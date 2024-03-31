import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,pipeline
from torch.nn.utils.rnn import pad_sequence
import concurrent.futures
from copy import deepcopy
from utils import HF_generate,extract_str_in_bracket,clean_answer,if_instruction_tuned,format_response,format_fs_qa,openai_call
from templates import self_reflection_prompt,self_answer_prompt
from collections import defaultdict
from functools import partial
import random
from openai import OpenAI
import time
from typing import List
import os
import requests
import warnings
from newspaper import Article
from tqdm import tqdm
from huggingface_hub import InferenceClient
import spacy

def async_process(fn,inps,workers=10,msg=''):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        out = list(tqdm(executor.map(fn,inps),total = len(inps),desc = msg))
    return out

NLI_MODEL = {'semantic_consistency':"microsoft/deberta-large-mnli",
             'BSDetector':"potsawee/deberta-v3-large-mnli",
             "SelfCheckGPT":"potsawee/deberta-v3-large-mnli"}

class NLIScorer:
    def __init__(self, gen_model,gen_model_name,tokenizer,scoring_method,beta=0.7,max_response_tokens= 128,answer_generator='self',use_tgi=True,answer_generator_port=8083,ref_as_chosen=False):
        """
        gen_model is the generation model
        beta is only used for BSDetector, weightage for self-reflection and Consistency score
        """
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL[scoring_method]).cuda()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL[scoring_method])
        self.gen_tokenizer = tokenizer
        self.gen_tokenizer.padding_side = 'left' #pad left side for generative.
        self.gen_model_name = gen_model_name
        self.scoring_method = scoring_method
        self.nli_model.eval()   
        self.gen_model = gen_model 
        self.beta = beta
        self.answer_generator = answer_generator
        if self.answer_generator == 'oracle':
            self.ans_generator = QAScorer(with_summarize=False)
        elif self.answer_generator == 'mistral_8x7': # uses another TGI
            self.ans_generator = InferenceClient(model = f"http://127.0.0.1:{answer_generator_port}")
            self.ans_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        self.use_tgi = use_tgi
        if not self.use_tgi:
            self.gen_model.eval()
        self.max_response_tokens = max_response_tokens
        self.max_nli_size = 100
        self.ref_as_chosen = ref_as_chosen # if reference answer is golden.
        self.nli_sentence_level = True 
        self.sentence_processor =  spacy.load("en_core_web_sm")
        
    def generate(self,prompt,mode = 'self_reflect'):
        do_sample=False
        if mode == 'self_reflect':
            max_new_tokens = 3
            repetition_penalty = 1.1
        else:
            max_new_tokens = self.max_response_tokens
            repetition_penalty = 1.1
        hf_gen_kwargs = {'do_sample':do_sample,'max_new_tokens':max_new_tokens,'repetition_penalty':repetition_penalty}
        out = HF_generate(prompt,self.gen_model,self.gen_tokenizer,hf_gen_kwargs,self.use_tgi,max_workers = len(prompt),return_as_dict=False)
        return out
    
    def get_nli_score(self,batch,return_full_score = False):
        """
        Assume batch contains a list of tuples where the 1st item is the reference sentence, while the 2nd item is a list of sampled sentences.
        """
        if self.nli_sentence_level:
            sentence_count = []
            full_batch = []
            for b in batch:
                sampled_ = b[1]
                sents = [s.text.strip() for s in self.sentence_processor(b[0]).sents]
                full_batch.extend([(s,sampled_) for s in sents])
                sentence_count.append(len(sents))
            assert sum(sentence_count) == len(full_batch), 'Sentence count does not match'
            all_preds = []
            for batch_id in range(0,len(full_batch),self.max_nli_size):
                batch = full_batch[batch_id:batch_id+self.max_nli_size]
                tokenized_inputs = self.nli_tokenizer.batch_encode_plus(batch_text_or_text_pairs = batch, padding='longest', return_tensors="pt",max_length = 512, truncation=True,add_special_tokens=True,return_token_type_ids=True, return_attention_mask=True)
                tokenized_inputs = {k: v.cuda() for k, v in tokenized_inputs.items()}
                with torch.no_grad():
                    nli_preds = self.nli_model(**tokenized_inputs).logits.detach().cpu()
                all_preds.append(nli_preds)
            all_preds = torch.cat(all_preds,dim=0)
            all_probs = torch.nn.functional.softmax(all_preds, dim=1)
            
            nli_probs = []
            ## split the full batch back to [list of individual samples where each list contains the sentence level preds]
            for num_sen in sentence_count:
                nli_probs.append(all_probs[:num_sen].mean(dim=0))
                all_probs = all_probs[num_sen:]
            nli_probs = torch.stack(nli_probs)
        else:
            tokenized_inputs = self.nli_tokenizer.batch_encode_plus(batch_text_or_text_pairs = batch, padding='longest', return_tensors="pt",max_length = 512, truncation=True,add_special_tokens=True,return_token_type_ids=True, return_attention_mask=True)
            tokenized_inputs = {k: v.cuda() for k, v in tokenized_inputs.items()}
            with torch.no_grad():
                nli_preds = self.nli_model(**tokenized_inputs).logits.detach().cpu()
            nli_probs = torch.nn.functional.softmax(nli_preds, dim=1)
            
        if not return_full_score:
            if self.scoring_method == 'BSDetector':
                return nli_probs[:,0] # entailment prob
            elif self.scoring_method == 'SelfCheckGPT':
                return nli_probs[:,1] # contradiction prob
        else:
            return nli_probs
        
    def get_score(self,instruction,ref_answer,sample_answer): ## use to score questions
        if self.use_tgi:
            sample_text = [sample_answer.generated_text] + [s.generated_text for s in sample_answer.details.best_of_sequences]
            ref_text = ref_answer.generated_text
        else:
            sample_text = [s['text'] for s in sample_answer]
            ref_text = ref_answer['text']
        if ref_text.strip() == '':
            return None
        all_answers = [ref_text]+ sample_text
        # Clean the answers if not instruct tuned.
        if not if_instruction_tuned(self.gen_model_name):
            sample_text = [clean_answer(x) for x in sample_text]
            ref_text = clean_answer(ref_text)
        
        if self.scoring_method == 'BSDetector':
            """
            From the paper: QUANTIFYING UNCERTAINTY IN ANSWERS FROM ANY LANGUAGE MODEL AND ENHANCING THEIR TRUSTWORTHINESS
            1) Get observed consistency score = entailment prob between ref answer and sample answers
            2) Get self-reflection certainty score = certainty of the model in its own answer
            3) combined scored = beta * observed consistency score + (1 - beta) * self-reflection certainty score
            ** Note that this is similar to SelfCheck NLI approach, but includes a self-reflection score where the model's self confidence is accounted for as well. If beta = 1, then it is the same as SelfCheck NLI (except the fact we look at entail probs instead of thresholding to [0,0.5,1])
            """
            ## Get Consistency score O ##
            nli_batch = [(ref_text,a) for a in sample_text]
            if self.ref_as_chosen:
                nli_scores =  self.get_nli_score(nli_batch,return_full_score=True) # we want the contradict scores as well.
                o = nli_scores[:,0].mean().item()
            else:
                o = self.get_nli_score(nli_batch).mean().item()
                nli_scores = None
            ## Get self-reflection certainty score S ##
            if self.beta < 1: # if beta is 1, then we dont need to calculate self-reflection score
                if if_instruction_tuned(self.gen_model_name):
                    self_reflect_prompt = [format_response([{'role':'user',
                                                            'content':self_reflection_prompt.format(instruction = instruction, answer = a)}]
                                                        ,self.gen_model_name,self.gen_tokenizer,mode='answer') 
                                                        for a in  all_answers]
                else:
                    self_reflect_prompt = [self_reflection_prompt.format(instruction = instruction, answer = a) for a in all_answers]
                    
                reflect_ans = self.generate(self_reflect_prompt,mode = 'self_reflect')
                if isinstance(reflect_ans[0],list): # unroll the list
                    reflect_ans = sum(reflect_ans,[])
                reflect_scores = []

                for ra in reflect_ans:
                    ra = extract_str_in_bracket(ra).strip()
                    if 'a' in ra.lower():
                        reflect_scores.append(1)
                    elif 'b' in ra.lower():
                        reflect_scores.append(0)
                    else:
                        reflect_scores.append(0.5)
                s = np.mean(reflect_scores)
            else:
                s = 0.
                reflect_scores = [0. for _ in range(len(all_answers))]
            
            overall_confidence =  (self.beta * o) + ((1 - self.beta) * s) # confidence for response to question x
            
            return {'confidence':overall_confidence,'instruction':instruction,'ref_answer':ref_text,'sample_answer':sample_text,'self_reflect_score':reflect_scores,'consistency_score':o,'all_nli_scores':nli_scores}
        
        elif self.scoring_method == 'semantic_consistency':
            """
            From the paper: SEMANTIC UNCERTAINTY: LINGUISTIC INVARIANCES FOR UNCERTAINTY ESTIMATION IN NATURAL LANGUAGE GENERATION
            1) Perform semantic clustering: group answers with similar semantic meaning together (Use NLI model to get entailment prediction)
            2) Compute entropy = -p*log(p) for generated sequence (average logprobs along sequence length and within each cluster grp first) - Use length normalization
            3) Overall entropy = mean over logprob of each sequence
            Ideally, bigger clusters have lower entropy
            Clusters with lowest entropy can be used as chosen answer where a random answer within the cluster is chosen, and clusters with highest entropy can be used as rejected answer
            Overall entropy is used as uncertainty score.
            """
            main_answer = [ref_answer,sample_answer] # because main_answer tokens stored differently then the rest
            if self.use_tgi:
                ## Setup text to logprob dict ##
                text_2_logprob = defaultdict()
                for main in main_answer:
                    text_2_logprob[main.generated_text] = (np.sum([t.logprob for t in main.details.tokens])/len(main.details.tokens))
                for rest in sample_answer.details.best_of_sequences:
                    text_2_logprob[rest.generated_text] = (np.sum([t.logprob for t in rest.tokens])/len(rest.tokens))
            else:
                if any([np.any(s['logprobs']==np.inf) for s in sample_answer]):
                    print ('inf logprob present')
                    return None
                text_2_logprob = {s['text']:s['logprobs'].mean() for s in main_answer}
            semantic_clusters = self.cluster_semantic_sets(instruction,all_answers)
            all_cluster_entropies,overall_entropy = self.get_cluster_entropy(semantic_clusters,text_2_logprob)

            return {'entropy':overall_entropy,'instruction':instruction,'sample_answer':sample_text,'ref_answer':ref_text,'all_cluster_entropies':all_cluster_entropies,
                    'semantic_clusters':semantic_clusters}
        
        elif self.scoring_method == 'SelfCheckGPT':
            nli_batch = [(ref_text,a) for a in sample_text]
            all_hallu_scores =  self.get_nli_score(nli_batch)
            hallu_score = all_hallu_scores.mean().item()
            
            return {'hallucination':hallu_score,'instruction':instruction,'ref_answer':ref_text,'sample_answer':sample_text,'all_hallu_scores':all_hallu_scores}
    
    def get_dpo_sample(self,content_dict,few_shots=None): # use confidence/uncertainty measures to score responses
        """
        few_shots only available if fixed_ds is used, aka reference dataset is available, in here, only used for API, self is generated out of this fn.
        Output dict should consist of 
        1) prompt, chosen and rejected answer for DPO training
        2) question confidence/uncertainty score to assess the model's confidence/uncertainty in self-generated questions
        3) topics to do post-training question generated to compare with 2)
        4) pre_response, to compare post-trained response using GPT4 or any other metrics to judge capability.
        """
        sample_answers = content_dict['sample_answer']
        instruction = content_dict['instruction']
        topic = content_dict['topic']
        ref_answer = content_dict['ref_answer']
        num_sample = len(sample_answers)

        # Between different scoring types, first pick worst answer, if self-generate best answer, pick it via the heuristics, else use oracle/gpt
        if self.scoring_method == 'BSDetector': # Each sampled answer act as reference, pick best and worse instead of always picking the ref answer
            remaining_reflect_scores = deepcopy(content_dict['self_reflect_score'])[1:] # remove the ref answer score
            if not self.ref_as_chosen:
                all_answers = [ref_answer]+ sample_answers
                ref_ans_confidence = (self.beta * content_dict['consistency_score']) + ((1 - self.beta) * content_dict['self_reflect_score'][0])
                all_ans_confidence  = [ref_ans_confidence]
                for i,sampled_ans in enumerate(sample_answers): # each sampled_answer will act as the reference ans
                    curr_ans_list = [sample_answers[j] for j in range(len(sample_answers)) if j != i] + [ref_answer]
                    curr_nli_batch = [(sampled_ans,a) for a in curr_ans_list]
                    curr_nli_score = self.get_nli_score(curr_nli_batch).mean().item()
                    all_ans_confidence.append((self.beta * curr_nli_score) + ((1 - self.beta) * remaining_reflect_scores[i]))

            else: # we compare contradict scores against ref answer to choose the worse answer. chosen is ref.
                all_ans_confidence = []
                all_answers = sample_answers
                if 'all_nli_scores' not in  content_dict:
                    content_dict['all_nli_scores'] = self.get_nli_score([ref_answer + ' [SEP] ' + a for a in sample_answers],return_full_score=True)
                contrad_scores = content_dict['all_nli_scores'][:,1].tolist()
                all_ans_confidence = [(self.beta * (1.0-contra)) + ((1 - self.beta) * curr_r) for contra,curr_r in zip(contrad_scores,remaining_reflect_scores)] # take 1-contra to convert to positive to check for min.
                
            max_confidence_id = np.argmax(all_ans_confidence)
            min_confidence_id = np.argmin(all_ans_confidence) 
            rejected_ans = all_answers[min_confidence_id]
            chosen_ans = all_answers[max_confidence_id] 
            question_score = 1.0 - content_dict['confidence'] # take the inverse of confidence

        elif self.scoring_method == 'semantic_consistency':
            semantic_clusters = content_dict['semantic_clusters'] # id to list of responses
            all_cluster_entropies = content_dict['all_cluster_entropies'] # id to entropy
            if self.ref_as_chosen: # if reference picked as chosen, remove the cluster that contains ref ans (since that is the chosen) and sample from rest.
                assert len(list(all_cluster_entropies.keys())) > 1, 'Only 1 cluster found, cannot sample'
                for cluster_id,cluster in semantic_clusters.items():
                    if ref_answer in cluster:
                        del semantic_clusters[cluster_id]
                        del all_cluster_entropies[cluster_id]
                        break
            sorted_cluster_entropies = {k:v for k,v in sorted(all_cluster_entropies.items(),key=lambda x:x[1])}
            rejected_ans = random.choice(semantic_clusters[list(sorted_cluster_entropies.keys())[-1]]) # highest entropy as rejected.
            chosen_ans = random.choice(semantic_clusters[list(sorted_cluster_entropies.keys())[0]]) # lowest entropy as chosen
            question_score = content_dict['entropy']
        
        elif self.scoring_method == 'SelfCheckGPT': ## Very similar to BSDetector but without self-reflect.
            if not self.ref_as_chosen:
                all_answers = [ref_answer]+ sample_answers
                all_hallu_scores = [content_dict['hallucination']]
                for i,sampled_ans in enumerate(sample_answers): 
                    curr_ans_list = [sample_answers[j] for j in range(len(sample_answers)) if j != i] + [ref_answer]
                    curr_nli_batch = [(sampled_ans,a) for a in curr_ans_list]
                    curr_nli_score = self.get_nli_score(curr_nli_batch).mean().item()
                    all_hallu_scores.append(curr_nli_score)
            else:
                all_answers = sample_answers
                all_hallu_scores = content_dict['all_hallu_scores']

            max_hallu_id = np.argmax(all_hallu_scores)
            min_hallu_id = np.argmin(all_hallu_scores)
            chosen_ans = all_answers[min_hallu_id]
            rejected_ans = all_answers[max_hallu_id]
            question_score = content_dict['hallucination']
            
        else:
            raise ValueError('Invalid scoring method')
        
        ## Few shot only used for external LLM such as GPT3.5/4 or Mistral_MOE
        if few_shots is not None:
            few_shots = few_shots[topic]
            fs_messages = format_fs_qa(few_shots,True)
        else:
            fs_messages = []
        
        if self.ref_as_chosen and 'self' in self.answer_generator: # force chosen ans as reference answer
            chosen_ans = ref_answer
            
        out_dict = {
                'instruction':instruction,
                'topic':topic,
                'chosen_ans':chosen_ans,
                'rejected_ans':rejected_ans,
                'question_score':question_score
                }
        ## Different answer generator ## if self just return out_dict else use another operator to get chosen answer.
        
        if self.answer_generator == 'oracle': # Use Google search to find documents and self-generate answer with document
            """
            Creating the chosen answer as "{extracted answer}. {summary}" seems to do poorly, as often, the extracted answer does not reliably answer the question.
            """
            scored_documents = self.ans_generator.get_scored_document(instruction) # returns a list of dicts containing document and score
            if len(scored_documents) == 0: # no valid documents
                return None
            if self.ans_generator.summarizer is None: # if we dont use summarized document as context answer, we generate our own.
                formatted_documents_w_score = [(self_answer_prompt.format(question = instruction,document = d['document']),d['score']) for d in scored_documents]
                document_lens = [len(self.gen_tokenizer.encode(x[0])) for x in formatted_documents_w_score]
                valid_idx = [i for i,x in enumerate(document_lens) if x < 3500] # filter out those with len > 3500
                if len(valid_idx) == 0:
                    return None
                formatted_documents_w_score = [formatted_documents_w_score[i] for i in valid_idx]
                selected_prompt = sorted(formatted_documents_w_score,key=lambda x:x[1],reverse=True)[0][0]
                chosen_ans = self.generate(selected_prompt,mode = 'answering')
                if isinstance(chosen_ans,list):
                    chosen_ans = chosen_ans[0]
                out_dict['chosen_ans'] = chosen_ans
            else:
                top_document = sorted(scored_documents,key=lambda x:x['score'],reverse=True)[0]
                selected_docu = top_document['document']
                selected_ans = top_document['answer']
                
                summarized_docu = self.ans_generator.summarizer(
                    selected_docu,
                    min_length=1,
                    max_length=self.max_response_tokens - len(self.gen_tokenizer.encode(selected_ans)),
                    truncation=True,
                    clean_up_tokenization_spaces=True
                )
                out_dict['chosen_ans'] = selected_ans + '. ' + summarized_docu[0]['summary_text']
            
            out_dict['link'] = [x['link'] for x in scored_documents]

        elif self.answer_generator in ['gpt4','gpt3.5']: # use OPENAI GPT3.5/4 to generate answers
            messages = [{'role':'user','content':instruction}]
            final_message = fs_messages + messages
            engine = 'gpt-4-0125-preview' if 'gpt4' in self.answer_generator else 'gpt-3.5-turbo-0125'
            openai_ans = openai_call(engine,final_message,self.max_response_tokens)
            if openai_ans is None:
                return None
            else:
                out_dict['chosen_ans'] = openai_ans

        elif self.answer_generator == 'mistral_8x7': ## Use Mistral 8x7b
            messages = [{'role':'user','content':instruction}]
            final_message = fs_messages + messages
            formatted_message = self.ans_tokenizer.apply_chat_template(final_message,tokenize=False,add_generation_prompt=True)
            ans_gen_kwargs = {'max_new_tokens':self.max_response_tokens,
                              'do_sample':False,
                              }
            response = self.ans_generator.text_generation(formatted_message,**ans_gen_kwargs)
            if '</s>' in response:
                response = response.split('</s>')[0].strip()
            out_dict['chosen_ans'] = response
            
        return out_dict
        
    def cluster_semantic_sets(self,instruction,all_responses):
        """
        Developed for semantic_consistency scoring method
        Cluster responses according to bidirectional entailment
        return cluster of responses
        """
        cluster_set = defaultdict(list)
        cluster_set[0] = [all_responses[0]]
        current_set_id = 1
        for response in all_responses[1:]:
            cluster_comparsions = defaultdict(list) # we compile all the comparsions for each cluster for batching
            for cluster_id,cluster in cluster_set.items():
                cluster_response = cluster[0]
                cluster_response_inp = instruction + ' ' + cluster_response
                curr_response_inp = instruction + ' ' + response
                nli_inp = cluster_response_inp + ' [SEP] ' + curr_response_inp
                reverse_nli_inp = curr_response_inp + ' [SEP] ' + cluster_response_inp
                cluster_comparsions[cluster_id].extend([nli_inp,reverse_nli_inp])
            
            all_cluster_comparsions = sum(list(cluster_comparsions.values()),[]) # flatten the list
            tokenized_nli_inp = self.nli_tokenizer(all_cluster_comparsions, padding=True, return_tensors="pt",max_length = 512, truncation=True)
            tokenized_nli_inp = {k: v.to(self.nli_model.device) for k, v in tokenized_nli_inp.items()}
            with torch.no_grad():
                pred = self.nli_model(**tokenized_nli_inp).logits.detach().cpu()
            pred_probs = torch.softmax(pred,dim=1)
            pred_label = torch.argmax(pred_probs,dim=1).numpy()
            entailed_probs = pred_probs[:,2].numpy().reshape(-1,2).mean(axis=1) # mean the probs of entailment
            pred_label = pred_label.reshape(-1,2) # reshape to get reverse and forward
            semantically_different = np.any(pred_label == 0,axis=1) # if any is contradiction, then it is semantically different
            semantically_similar = ~semantically_different
            
            if np.all(semantically_different): # if all are semantically different, then create a new cluster
                cluster_set[current_set_id] = [response]
                current_set_id+=1
            else: # they is a cluster with similar meaning
                cluster_ids = list(cluster_comparsions.keys())
                all_entailed_probs = entailed_probs[semantically_similar]
                all_entailed_ids = (np.array(cluster_ids)[semantically_similar]).tolist()
                max_entailed_id = all_entailed_ids[np.argmax(all_entailed_probs)] # get the cluster id with the highest entailed prob
                cluster_set[max_entailed_id].append(response)
            
        return cluster_set
    
    def get_cluster_entropy(self,semantic_cluster,responses_logprobs):
        """
        1) For each cluster, get the logprobs of each response, sum up across sequence to get joint logprobs
        2) Compute entropy within each cluster = - log(sum(p(s|x))) * sum(p(s|x)) = entropy over cluster meaning rather than individal seq
        3) Compute overall entropy = mean across each cluster logprobs, author use monte carlo integration, which is 1/C * sum(log p(C|x))
        
        responses_logprobs is a dict where key is the response and value is the total joint logprobs
        semantic_cluster is a dict of lists where key is the cluster id and value is a list of responses
        Return -> Return both cluster and overall entropy
        """
        overall_entropies = []
        all_cluster_entropies = {}
        for semantic_id,semantic_responses in semantic_cluster.items():
            cluster_logprobs = []
            for response in semantic_responses:
                joint_logprobs = responses_logprobs[response]
                cluster_logprobs.append(joint_logprobs) # convert to higher precision
            cluster_logprobs = torch.logsumexp(cluster_logprobs,dim=0) # sum logprobs over all responses within the cluster = log(sum(p(s|x)))
            overall_entropies.append(cluster_logprobs)
            
            cluster_entropy = -(cluster_logprobs * torch.exp(cluster_logprobs)) # - log(sum(p(s|x))) * sum(p(s|x))
            all_cluster_entropies[semantic_id] = cluster_entropy.item()
        
        overall_entropies = - torch.mean(torch.stack(overall_entropies)).item()
        
        return all_cluster_entropies,overall_entropies

class QAScorer(): # to score documents using a QA answering system
    def __init__(
            self,
            with_summarize = True
        ) -> None:
        self.max_batch_size = 8
        self.question_answerer = pipeline(
            "question-answering", tokenizer="deepset/tinyroberta-squad2",
            model="deepset/tinyroberta-squad2", framework="pt",batch_size = self.max_batch_size,
            device=0
        )
        if with_summarize:
            self.summarizer = pipeline(
            "summarization", tokenizer="facebook/bart-large-cnn",
            model="facebook/bart-large-cnn", framework="pt",batch_size = self.max_batch_size,
            device=0
            )
        else:
            self.summarizer = None
    
    def search_engine_fn(self,query: str) -> List[str]:
        if "GOOGLE_CUSTOM_SEARCH_URL" not in os.environ:
            raise ValueError("The environment variable GOOGLE_CUSTOM_SEARCH_URL is not set!")
        try:
            url = str(os.environ.get("GOOGLE_CUSTOM_SEARCH_URL")) + query
            response = requests.get(url)
            result = response.json()
            return [x["link"] for x in result["items"]]
        except Exception as e:
            warnings.warn("Error when searching on Google. | Error: " + str(e))
            pass
        return []

    def score_document(self, question: str, links: str):
        all_inps = [{'link':l} for l in links]
        all_inps = async_process(self.get_documents,all_inps,workers = 16)
        success_documents = [x for x in all_inps if x['document'] is not None]
        out = []
        for i in range(0, len(success_documents), self.max_batch_size):
            current_batch = success_documents[i:i+self.max_batch_size]
            qa_output = self.question_answerer(
                question=[question for _ in current_batch], context=[x['document'] for x in current_batch]
            )
            for o,qa in zip(current_batch,qa_output):
                if isinstance(o,dict) and isinstance(qa,dict): # in some cases qa_output returns a str, error?
                    o['answer'] = qa['answer']
                    o['score'] = qa['score']
                else:
                    o['score'] = -1
                    o['answer'] = None
            out.extend(current_batch)
        out = [o for o in out if o['answer'] is not None]
        return out
    
    def get_documents(self,link_dict):
        link = link_dict['link']
        try:
            article = Article(link)
            article.download()
            article.parse()
            article.nlp()
            if not article.text:
                full_text =  None
            else:
                full_text =  article.text
        except Exception as e:
            full_text =  None
        link_dict['document'] = full_text
        return link_dict

    def get_scored_document(self, question):
        links = self.search_engine_fn(question)
        if len(links)== 0:
            return []
        scored_documents = self.score_document(question, links)
        return scored_documents

class LLMJudge(): # use GPT3.5/4 to judge 2 given responses
    def __init__(self,engine = 'gpt-4-0125-preview'):
        self.client = OpenAI()
        self.eval_prompt = [{'role':'system','content':'You are a unbiased judge who is evaluating a set of response to a given question.'},
                            {'role':'user','content':'Question: {question}\n Response A: {response_a}\nResponse B: {response_b}\nWhich response is better in answering the question? You are to answer with either A or B only.\nAnswer: '}]
        self.engine = engine
        self.max_concurrent_calls = 16
        self.max_response_tokens = 1
    
    def get_openai_choice(self,messages):
        message = messages['message']
        chosen_choice = messages['chosen_choice']
        instr = messages['instruction']
        response = None
        try:
            response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=message,
                    temperature=0,
                    max_tokens=self.max_response_tokens,
                    )
        except Exception as e:
            time.sleep(2)
            response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=message,
                    temperature=0,
                    max_tokens=self.max_response_tokens,
                    )
        if response is None:
            return {'result':None,
                    'instr': instr,
                    'in':0,
                    'out':0}
        resp =  response.choices[0].message.content
        inp_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
        if resp.lower() == chosen_choice.lower(): # check if chosen is same as the post response
            result = 1
        else:
            result = 0
        return {'result':result,
                'instr': instr,
                'in':inp_tokens,
                'out':out_tokens}
        
    def evaluate(self,content_dicts):
        all_messages = []
        for content_dict in content_dicts:
            instruction = content_dict['instruction']
            pre_response = content_dict['pre_response']
            post_response = content_dict['post_response']
            random_chosen = random.choice(['A','B']) # randomize the choices
            if random_chosen == 'A':
                response_a = post_response
                response_b = pre_response
            else:
                response_a = pre_response
                response_b = post_response
                
            curr_content = deepcopy(self.eval_prompt)
            curr_content[1]['content'] = curr_content[1]['content'].format(question = instruction,response_a = response_a,response_b = response_b)
            all_messages.append({'message':curr_content,'chosen_choice':random_chosen,'instruction':instruction})
        
        results = async_process(self.get_openai_choice,all_messages,self.max_concurrent_calls)
        results = [r for r in results if r['result'] is not None]
        instr2_score = {}
        for r in results:
            instr2_score[r['instr']] = r['result']
        mean_score = sum(list(instr2_score.values()))/len(list(instr2_score.values()))
        total_in_tokens = sum([r['in'] for r in results])
        total_out_tokens = sum([r['out'] for r in results])
        
        # compute cost
        if 'gpt-4' in self.engine:
            cost = 0.01 * (total_in_tokens/1000) + 0.03 * (total_out_tokens/1000)
        else:
            cost = 0.0005 * (total_in_tokens/1000) + 0.0015 * (total_out_tokens/1000)
        
        return mean_score,instr2_score,round(cost,3)
        
        
            
        
        
                
        
        
        
            
      
                            
            
            
            