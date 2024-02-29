import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,pipeline
from torch.nn.utils.rnn import pad_sequence
import concurrent.futures
from copy import deepcopy
from utils import HF_generate,extract_str_in_bracket
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

def async_process(fn,inps,workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        out = list(executor.map(fn,inps))
    return out

class NLIScorer:
    def __init__(self, gen_model,model_path,tokenizer,scoring_method,beta=0.7,max_response_tokens= 128,answer_generator='self',use_tgi=True,prompt_fn_dict=None):
        """
        gen_model is the generation model
        beta is only used for BSDetector, weightage for self-reflection and Consistency score
        """
        print ('model_path')
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gen_tokenizer = tokenizer
        self.scoring_method = scoring_method
        self.nli_model.eval()   
        self.gen_model = gen_model 
        self.beta = beta
        self.answer_generator = answer_generator
        if self.answer_generator == 'oracle':
            self.ans_generator = QAScorer()
        self.use_tgi = use_tgi
        if not self.use_tgi:
            self.gen_model.eval()
        self.extract_ans_fn = prompt_fn_dict['extract_ans_fn']
        self.prompt_fn = prompt_fn_dict['prompt_fn']['answer_gen']
        self.max_response_tokens = max_response_tokens
        
    def generate(self,prompt,mode = 'self_reflect'):
        do_sample=False
        if mode == 'self_reflect':
            max_new_tokens = 3
            repetition_penalty = 1.0
        else:
            max_new_tokens = self.max_response_tokens
            repetition_penalty = 1.1
        if self.use_tgi:
            return self.gen_model.text_generation(prompt, max_new_tokens=max_new_tokens, do_sample = do_sample, repetition_penalty = repetition_penalty)
        else:
            hf_gen_kwargs = {'do_sample':do_sample,'max_new_tokens':max_new_tokens,'repetition_penalty':repetition_penalty}
            out =  HF_generate(prompt,self.gen_model,self.gen_tokenizer,hf_gen_kwargs,self.extract_ans_fn)
            return [o['text'] for o in out]
            
    
    def get_nli_score(self,ref_answer,sample_answers):
        all_concatenated_ans = [ref_answer + ' [SEP] ' + sample_a for sample_a in sample_answers] 
        tokenized_inputs = self.nli_tokenizer(all_concatenated_ans, padding=True, return_tensors="pt",max_length = 512, truncation=True)
        tokenized_inputs = {k: v.cuda() for k, v in tokenized_inputs.items()}
        with torch.no_grad():
            nli_preds = self.nli_model(**tokenized_inputs).logits
        nli_probs = torch.nn.functional.softmax(nli_preds, dim=1)
        entailment_probs = nli_probs[:,2]
        mean_entailment_probs = torch.mean(entailment_probs).item()
        
        return mean_entailment_probs

    def get_score(self,instruction,ref_answer,sample_answer): ## use to score questions
        if self.use_tgi:
            sample_text = [sample_answer.generated_text] + [s.generated_text for s in sample_answer.details.best_of_sequences]
            if ref_answer is not None:
                ref_answer = ref_answer.generated_text
        else:
            sample_text = [s['text'] for s in sample_answer]
            if ref_answer is not None:
                ref_answer = ref_answer['text'] 
        if self.scoring_method == 'BSDetector':
            """
            From the paper: QUANTIFYING UNCERTAINTY IN ANSWERS FROM ANY LANGUAGE MODEL AND ENHANCING THEIR TRUSTWORTHINESS
            1) Get observed consistency score = entailment prob between ref answer and sample answers
            2) Get self-reflection certainty score = certainty of the model in its own answer
            3) combined scored = beta * observed consistency score + (1 - beta) * self-reflection certainty score
            ** Note that this is similar to SelfCheck NLI approach, but includes a self-reflection score where the model's self confidence is accounted for as well. If beta = 1, then it is the same as SelfCheck NLI (except the fact we look at entail probs instead of thresholding to [0,0.5,1])
            """
            ## Get Consistency score O ##
            all_answers = [ref_answer]+ sample_text
            o = self.get_nli_score(ref_answer,sample_text)
            ## Get self-reflection certainty score S ##
            if self.beta < 1: # if beta is 1, then we dont need to calculate self-reflection score
                self_reflect_prompt = [self.prompt_fn(self_reflection_prompt.format(instruction = instruction, answer = a)) for a in  all_answers]
                if self.use_tgi:
                    reflect_fn = partial(self.generate,mode = 'self_reflect')
                    reflect_ans = async_process(reflect_fn,self_reflect_prompt,workers = len(self_reflect_prompt))
                else:
                    reflect_ans = self.generate(self_reflect_prompt,mode = 'self_reflect')
                if isinstance(reflect_ans[0],list): # unroll the list
                    reflect_ans = sum(reflect_ans,[])
                reflect_scores = []
                reflect_logs = defaultdict(int)
                for ra in reflect_ans:
                    ra = extract_str_in_bracket(ra)
                    if 'a' in ra.lower():
                        reflect_logs['a']+=1
                        reflect_scores.append(1)
                    elif 'b' in ra.lower():
                        reflect_logs['b']+=1
                        reflect_scores.append(0)
                    else:
                        reflect_logs['c']+=1
                        reflect_scores.append(0.5)
                s = np.mean(reflect_scores)
            else:
                s = 0.
                reflect_scores = [0. for _ in range(len(all_answers))]
                reflect_logs=None
            
            overall_confidence =  (self.beta * o) + ((1 - self.beta) * s) # confidence for response to question x
            
            return {'confidence':overall_confidence,'instruction':instruction,'ref_answer':ref_answer,'sample_answer':sample_text,'self_reflect_score':reflect_scores,'consistency_score':o,'reflect_logs':reflect_logs}
        
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
            if self.use_tgi:
                ## Setup text to logprob dict ##
                text_2_logprob = defaultdict()
                main_answer = sample_text[0]
                rest_answer = sample_text[1:]
                text_2_logprob[main_answer] = (np.sum([t.logprob for t in sample_answer.details.tokens])/len(sample_answer.details.tokens)).item()
                for rest_text,rest_logp in zip(rest_answer,sample_answer.details.best_of_sequences):
                    text_2_logprob[rest_text] = (np.sum([t.logprob for t in rest_logp.tokens])/len(rest_logp.tokens)).item()
            else:
                if any([np.any(s['logprobs']==np.inf) for s in sample_answer]):
                    print ('inf logprob present')
                    return None
                text_2_logprob = {s['text']:s['logprobs'].mean().item() for s in sample_answer}
            semantic_clusters = self.cluster_semantic_sets(instruction,sample_text)
            all_cluster_entropies,overall_entropy = self.get_cluster_entropy(semantic_clusters,text_2_logprob)

            return {'overall_entropy':overall_entropy,'instruction':instruction,'sample_answer':sample_text,'ref_answer':ref_answer,'all_cluster_entropies':all_cluster_entropies,
                    'semantic_clusters':semantic_clusters}
    
    def get_dpo_sample(self,content_dict): # use confidence/uncertainty measures to score responses
        """
        Output dict should consist of 
        1) prompt, chosen and rejected answer for DPO training
        2) question confidence/uncertainty score to assess the model's confidence/uncertainty in self-generated questions
        3) topics to do post-training question generated to compare with 2)
        4) pre_response, to compare post-trained response using GPT4 or any other metrics to judge capability.
        """
        sample_answers = content_dict['sample_answer']
        instruction = content_dict['instruction']
        pre_response = content_dict['ref_answer']
        if not isinstance(pre_response,str):
            pre_response = pre_response.generated_text
        # Between different scoring types, first pick worst answer, if self-generate best answer, pick it via the heuristics, else use oracle/gpt
        if self.scoring_method == 'BSDetector':
            ref_answer = content_dict['ref_answer']
            qn_metric = {'qn_confidence':content_dict['confidence']}
        elif self.scoring_method == 'semantic_consistency':
            semantic_clusters = content_dict['semantic_clusters'] # id to list of responses
            all_cluster_entropies = content_dict['all_cluster_entropies'] # id to entropy
            sorted_cluster_entropies = {k:v for k,v in sorted(all_cluster_entropies.items(),key=lambda x:x[1])}
            ref_answer = random.choice(semantic_clusters[list(sorted_cluster_entropies.keys())[-1]]) # pick the answer with the highest entropy
            qn_metric = {'qn_entropy':content_dict['overall_entropy']}
        else:
            raise ValueError('Invalid scoring method')
            
        if self.answer_generator == 'self': # pick either most confident or least entropy response as chosen answer #
            if self.scoring_method == 'BSDetector':
                all_answers = [ref_answer]+ sample_answers
                ref_ans_confidence = (self.beta * content_dict['consistency_score']) + ((1 - self.beta) * content_dict['self_reflect_score'][0])
                
                remaining_reflect_scores = deepcopy(content_dict['self_reflect_score'])[1:] # remove the ref answer score
                all_ans_confidence  = [ref_ans_confidence]
                for i,sampled_ans in enumerate(sample_answers): # each sampled_answer will act as the reference ans
                    curr_ans_list = [a for a in deepcopy(sample_answers) if a != sampled_ans] + [ref_answer]
                    curr_o = self.get_nli_score(sampled_ans,curr_ans_list)
                    all_ans_confidence.append((self.beta * curr_o) + ((1 - self.beta) * remaining_reflect_scores[i]))
                
                max_confidence_id = np.argmax(all_ans_confidence)
                min_confidence_id = np.argmin(all_ans_confidence) 
                
                chosen_ans = all_answers[max_confidence_id]
                rejected_ans = all_answers[min_confidence_id]
                max_confidence = all_ans_confidence[max_confidence_id]
                min_confidence = all_ans_confidence[min_confidence_id]
                
                return {**qn_metric,
                    'chosen_ans':chosen_ans,
                    'rejected_ans':rejected_ans,
                    'max_confidence':max_confidence,
                    'min_confidence':min_confidence,
                    'instruction':instruction,
                    'topic':content_dict['topic'],
                    'pre_response':pre_response
                    }
                
            elif self.scoring_method == 'semantic_consistency':
                chosen_ans = random.choice(semantic_clusters[list(sorted_cluster_entropies.keys())[0]])
                rejected_ans = ref_answer
                return {**qn_metric,'chosen_ans':chosen_ans,
                    'rejected_ans':rejected_ans,
                    'chosen_entropy':list(sorted_cluster_entropies.values())[0],
                    'rejected_entropy':list(sorted_cluster_entropies.values())[-1],
                    'instruction':instruction,
                    'topic':content_dict['topic'],
                    'pre_response':pre_response
                    }    
            
        elif self.answer_generator == 'oracle': # Use Google search to find documents and self-generate answer with document
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
                chosen_ans = selected_ans + '. ' + summarized_docu[0]['summary_text']
                
            # rejected_ans = random.choice(sample_answers) # randomly sample from the sample answers
            rejected_ans = ref_answer
            
            return {**qn_metric,'chosen_ans':chosen_ans,
                    'rejected_ans':rejected_ans,
                    'instruction':instruction,
                    'topic':content_dict['topic'],
                    'link': [x['link'] for x in scored_documents],
                    'pre_response':pre_response
                    }
        
        elif 'gpt' in self.answer_generator: # use OPENAI GPT3.5/4 to generate answers
            client = OpenAI()
            messages = [{'role':'user','content':instruction}]
            engine = 'gpt-4-0125-preview' if 'gpt4' in self.answer_generator else 'gpt-3.5-turbo-0125'
            try:
                response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0,
                max_tokens=self.max_response_tokens,
                )
            except Exception as e:
                time.sleep(2)
                response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0,
                max_tokens=128,
                )
            chosen_ans = response.choices[0].message.content
            rejected_ans = ref_answer
            inp_tokens = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
            return {**qn_metric,'chosen_ans':chosen_ans,
                    'rejected_ans':rejected_ans,
                    'instruction':instruction,
                    'topic':content_dict['topic'],
                    'inp_tokens':inp_tokens,
                    'out_tokens':out_tokens,
                    'pre_response':pre_response
                    }
        
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
            entailed = np.all(pred_label == 2,axis=1) # if both directions are entailment, then it is entailed
            
            cluster_ids = list(cluster_comparsions.keys())
            all_entailed_probs = entailed_probs[entailed]
            all_entailed_ids = (np.array(cluster_ids)[entailed]).tolist()
            if len(all_entailed_probs) > 0:
                max_entailed_id = all_entailed_ids[np.argmax(all_entailed_probs)] # get the cluster id with the highest entailed prob
                cluster_set[max_entailed_id].append(response)
            else:
                cluster_set[current_set_id] = [response]
                current_set_id+=1
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
                cluster_logprobs.append(torch.tensor(joint_logprobs,dtype = torch.float32)) # convert to higher precision
            cluster_logprobs = torch.logsumexp(torch.stack(cluster_logprobs),dim=0) # sum logprobs over all responses within the cluster = log(sum(p(s|x)))
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
        def get_documents(link_dict):
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
        
        all_inps = [{'link':l} for l in links]
        all_inps = async_process(get_documents,all_inps,workers = 16)
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
            all_messages.append({'message':curr_content,'chosen_choice':random_chosen})
        
        results = async_process(self.get_openai_choice,all_messages,self.max_concurrent_calls)
        scores = [r['result'] for r in results if r['result'] is not None]
        total_in_tokens = sum([r['in'] for r in results])
        total_out_tokens = sum([r['out'] for r in results])
        
        # compute cost
        if 'gpt-4' in self.engine:
            cost = 0.01 * (total_in_tokens/1000) + 0.03 * (total_out_tokens/1000)
        else:
            cost = 0.0005 * (total_in_tokens/1000) + 0.0015 * (total_out_tokens/1000)
        
        return sum(scores)/len(scores),round(cost,3)
        
        
            
        
        
                
        
        
        
            
      
                            
            
            
            