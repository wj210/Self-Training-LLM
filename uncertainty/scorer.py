import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,pipeline
from torch.nn.utils.rnn import pad_sequence
import concurrent.futures
from copy import deepcopy
from utils import *
from templates import self_reflection_prompt
from collections import defaultdict
import random
from openai import OpenAI
import time
from typing import List
from tqdm import tqdm
from huggingface_hub import InferenceClient
import spacy
import string
import os
import pickle

def async_process(fn,inps,workers=10,msg=''):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        out = list(tqdm(executor.map(fn,inps),total = len(inps),desc = msg))
    return out

NLI_MODEL = {'semantic_consistency':"microsoft/deberta-large-mnli",
             'BSDetector':"potsawee/deberta-v3-large-mnli",
             "SelfCheckGPT":"potsawee/deberta-v3-large-mnli"}

class NLIScorer:
    def __init__(self, gen_model,gen_model_name,tokenizer,scoring_method,beta=0.7,max_response_tokens= 128,answer_generator='self',use_tgi=True,ref_as_chosen=True):
        """
        gen_model is the generation model
        beta is only used for BSDetector, weightage for self-reflection and Consistency score
        """
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL[scoring_method]).cuda()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL[scoring_method])
        self.gen_tokenizer = tokenizer
        if self.gen_tokenizer is not None:
            self.gen_tokenizer.padding_side = 'left' #pad left side for generative.
        self.gen_model_name = gen_model_name
        self.scoring_method = scoring_method
        self.nli_model.eval()   
        self.gen_model = gen_model 
        self.beta = beta
        self.answer_generator = answer_generator
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
        else:
            return nli_probs
        
    def get_score(self,instruction,ref_answer,sample_answer,base=False,compute_sample_scores=False): ## use to score questions
        if self.use_tgi:
            if not isinstance(sample_answer,list):
                sample_text = get_tgi_text(sample_answer)
                if not isinstance(sample_text,list): # TEMP
                    sample_text = [sample_text]
            else:
                sample_text = sample_answer
            if not isinstance(ref_answer,str):
                ref_text = get_tgi_text(ref_answer)
            else:
                ref_text = ref_answer
        else:
            sample_text = sample_answer['text']
            ref_text = ref_answer['text']

        
        
        ## Check for empty strings
        if base:
            sample_text = [clean_base_response(s) for s in sample_text] # clean sampled responsed, ref already cleaned
        sample_text = [s for s in sample_text if s.strip() != '']
        if ref_text.strip() == '' or len(sample_text) == 0:
            return None
        
        if self.ref_as_chosen:
            all_answers = [ref_text]+ sample_text
        else:
            all_answers = sample_text

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
            nli_scores =  self.get_nli_score(nli_batch,return_full_score=True) # we want the contradict scores as well.
            o = nli_scores[:,0].mean().item()

            ## Get self-reflection certainty score S ##
            if self.beta < 1: # if beta is 1, then we dont need to calculate self-reflection score
                self_reflect_prompt = [format_response([{'role':'user',
                                                        'content':self_reflection_prompt.format(instruction = instruction, answer = a)}]
                                                    ,self.gen_model_name,self.gen_tokenizer,mode='answer') 
                                                    for a in  all_answers]

                    
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

            return {'confidence':overall_confidence,'self_reflect_score':reflect_scores,'consistency_score':o,'all_nli_scores':nli_scores}

        
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

            main_answer = [sample_answer] # because main_answer tokens stored differently then the rest

            if self.use_tgi:
                ## Setup text to logprob dict ##
                text_2_logprob = defaultdict()
                for main in main_answer: # get the normalized sum logprobs
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

            return {'entropy':overall_entropy,'all_cluster_entropies':all_cluster_entropies,'semantic_clusters':semantic_clusters}
        
        elif self.scoring_method == 'SelfCheckGPT':
            if not (compute_sample_scores and not self.ref_as_chosen):
                nli_batch = [(ref_text,a) for a in sample_text]
                all_hallu_scores =  self.get_nli_score(nli_batch).tolist()
                hallu_score = np.mean(all_hallu_scores).item() # main ref hallu score
            
            # elif not self.ref_as_chosen and compute_sample_scores:
            else:
                sep_token = "[SEP]"
                hallu_score_dicts = {}
                remaining_nli_samples = []
                
                # all_hallu_scores = [hallu_score] # reset, store it as ref against all samples
                all_hallu_scores = []
                for i,sampled_ans in enumerate(sample_text): # Do this so that we dont do extra nli score calculation
                    curr_sample_list = [sample_text[j] for j in range(len(sample_text)) if j != i]
                    # curr_sample_list = [s for s in curr_sample_list if s != sampled_ans] # remove duplicates
                    check_key = [[sep_token.join([sa,sampled_ans]),sep_token.join([sampled_ans,sa])] for sa in curr_sample_list]
                    for curr_sample_ck in check_key:
                        if curr_sample_ck[0] not in hallu_score_dicts and curr_sample_ck[1] not in hallu_score_dicts:
                            hallu_score_dicts[curr_sample_ck[0]] = 0.  # Set placeholder value
                            remaining_nli_samples.append(tuple(curr_sample_ck[0].split(sep_token)))
                            
                if len(remaining_nli_samples) < 1:
                    return None
                all_nli_scores = self.get_nli_score(remaining_nli_samples)
                for nli_key,nli_score in zip(remaining_nli_samples,all_nli_scores): # record down keys
                    joined_nli_key = sep_token.join(nli_key)
                    hallu_score_dicts[joined_nli_key] = nli_score
                    
                for i,sampled_ans in enumerate(sample_text): # collate all the nli scores that each sampled ans is compared against the rest
                    sampled_nli_score = []
                    # curr_sample_list = [sample_text[j] for j in range(len(sample_text)) if j != i] + [ref_text]
                    curr_sample_list = [sample_text[j] for j in range(len(sample_text)) if j != i]
                    # curr_sample_list = [s for s in curr_sample_list if s != sampled_ans]
                    check_key = [[sep_token.join([sa,sampled_ans]),sep_token.join([sampled_ans,sa])] for sa in curr_sample_list]
                    for curr_sample_ck in check_key:
                        for ck in curr_sample_ck:
                            if ck in hallu_score_dicts:
                                sampled_nli_score.append(hallu_score_dicts[ck])
                                break

                    all_hallu_scores.append(np.mean(sampled_nli_score).item())
                
                hallu_score = np.mean(all_hallu_scores).item()
                if len(all_hallu_scores) != len(sample_text): # if doesnt match, then we have missing nli scores
                    return None

            return {'hallucination':hallu_score,'all_hallu_scores':all_hallu_scores}

    
    def get_dpo_sample(self,content_dict,multiple_pref=False,unknown_filtering=0.,question_filtering=1.0,greedy_as_rejected=False): # use confidence/uncertainty measures to score responses
        """
        fs_messages only for Stronger LLM usage
        Output dict should consist of 
        1) prompt, chosen and rejected answer for DPO training
        2) question confidence/uncertainty score to assess the model's confidence/uncertainty in self-generated questions
        3) topics to do post-training question generated to compare with 2)
        4) pre_response, to compare post-trained response using GPT4 or any other metrics to judge capability.
        """
        raw_answer_sample = get_tgi_text(content_dict['raw_answer_sample'])
        if not isinstance(raw_answer_sample,list): # TEMP
            raw_answer_sample = [raw_answer_sample]
        instruction = content_dict['instruction']
        topic = content_dict['topic']
        category = content_dict['category']
        if not isinstance(content_dict['gold_answer'],str):
            gold_answer = get_tgi_text(content_dict['gold_answer'])
        else:
            gold_answer = content_dict['gold_answer']
        question_score = content_dict[SCORE_KEY[self.scoring_method]]
        if self.scoring_method == 'BSDetector':
            question_score = 1. - question_score
        
        non_empty_raw_ans_pos = [i for i,r in enumerate(raw_answer_sample) if r.strip() != '']
        if len(raw_answer_sample) != len(content_dict['all_hallu_scores']):
            if multiple_pref:
                return [None]
            else:
                return None 
        raw_answer_sample = [raw_answer_sample[i] for i in non_empty_raw_ans_pos]
        
        if self.scoring_method == 'SelfCheckGPT':
            content_dict['all_hallu_scores'] = np.array([content_dict['all_hallu_scores'][i] for i in non_empty_raw_ans_pos])
        elif self.scoring_method == 'BSDetector':
            content_dict['all_nli_scores'] = np.array(content_dict['all_nli_scores'][non_empty_raw_ans_pos])
        
        if len(raw_answer_sample) == 0:
            if multiple_pref:
                return [None]
            else:
                return None
        
        ## For multiple pref
        gold_answer_sample = content_dict.get('gold_answer_sample',None)
        if gold_answer_sample is not None:
            if not isinstance(content_dict['gold_answer'],list):
                gold_answer_sample = get_tgi_text(gold_answer_sample)
            else:
                gold_answer_sample = content_dict['gold_answer_sample']

        if not multiple_pref: # if not multiple pref, we choose the most hallucinated/inconsistent sample from raw_answer_sample as rejected
            if self.scoring_method == 'BSDetector':
                remaining_reflect_scores = deepcopy(content_dict['self_reflect_score'])[1:] # remove the ref answer score
                if not self.ref_as_chosen:
                    all_answers = [gold_answer]+ raw_answer_sample
                    ref_ans_confidence = (self.beta * content_dict['consistency_score']) + ((1 - self.beta) * content_dict['self_reflect_score'][0])
                    all_ans_confidence  = [ref_ans_confidence]
                    for i,sampled_ans in enumerate(raw_answer_sample): # each sampled_answer will act as the reference ans
                        curr_ans_list = [raw_answer_sample[j] for j in range(len(raw_answer_sample)) if j != i] + [gold_answer]
                        curr_nli_batch = [(sampled_ans,a) for a in curr_ans_list]
                        curr_nli_score = self.get_nli_score(curr_nli_batch).mean().item()
                        all_ans_confidence.append((self.beta * curr_nli_score) + ((1 - self.beta) * remaining_reflect_scores[i]))

                else: # we compare contradict scores against ref answer to choose the worse answer. chosen is ref.
                    all_ans_confidence = []
                    all_answers = raw_answer_sample
                    contrad_scores = content_dict['all_nli_scores'][:,1].tolist()
                    all_ans_confidence = [(self.beta * (1.0-contra)) + ((1 - self.beta) * curr_r) for contra,curr_r in zip(contrad_scores,remaining_reflect_scores)] # take 1-contra to convert to positive to check for min.
                
                max_confidence_id = np.argmax(all_ans_confidence)
                min_confidence_id = np.argmin(all_ans_confidence) 
                rejected_ans = all_answers[min_confidence_id]
                chosen_ans = all_answers[max_confidence_id] 


            elif self.scoring_method == 'semantic_consistency':
                if not self.ref_as_chosen:
                    semantic_clusters = content_dict['semantic_clusters'] # id to list of responses
                    all_cluster_entropies = content_dict['all_cluster_entropies'] # id to entropy
                    sorted_cluster_entropies = {k:v for k,v in sorted(all_cluster_entropies.items(),key=lambda x:x[1])}
                    rejected_ans = random.choice(semantic_clusters[list(sorted_cluster_entropies.keys())[-1]]) # highest entropy as rejected.
                    chosen_ans = random.choice(semantic_clusters[list(sorted_cluster_entropies.keys())[0]]) # lowest entropy as chosen
            
            elif self.scoring_method == 'SelfCheckGPT': ## Very similar to BSDetector but without self-reflect.
                all_answers =  raw_answer_sample
                
                assert len(all_answers) == len(content_dict['all_hallu_scores']), f'{len(all_answers)} != {len(content_dict["all_hallu_scores"])}'
                all_hallu_scores = content_dict['all_hallu_scores']
                max_hallu_id = np.argmax(all_hallu_scores)
                min_hallu_id = np.argmin(all_hallu_scores)
                chosen_ans = all_answers[min_hallu_id]
                rejected_ans = all_answers[max_hallu_id]
                rejected_hallu_score = np.max(all_hallu_scores)
   
            else:
                raise ValueError('Invalid scoring method')
        
        if self.ref_as_chosen: # force chosen ans as reference answer
            chosen_ans = gold_answer
        
        if greedy_as_rejected and not multiple_pref:
            rejected_ans = get_tgi_text(content_dict['raw_answer'])
        
        if multiple_pref and self.ref_as_chosen:
            ## Further evalaute gold answer samples to check for empty strs or abnormal answers that largely contradicts greedy gold_answer
            gold_answer_sample_scores = content_dict['gold_answer_scores']
            if isinstance(gold_answer_sample_scores,torch.Tensor):
                gold_answer_sample_scores = gold_answer_sample_scores.numpy()
            elif isinstance(gold_answer_sample_scores,list):
                gold_answer_sample_scores = np.array(gold_answer_sample_scores)
            if question_filtering < 1.0: # if activated, filter out answers that are largely contradicted by greedy gold_answer
                selected_gold_pos = np.where(gold_answer_sample_scores < question_filtering)[0] # filter out answers that are largely contradicted by greedy gold_answer
            else:
                selected_gold_pos = range(len(gold_answer_sample))
            gold_answer_sample = [(gold_answer_sample[i],gold_answer_sample_scores[i]) for i in selected_gold_pos]
            
            
            if self.scoring_method == 'BSDetector': # take 1: as the first is the raw greedy.
                raw_answer_scores = content_dict['all_nli_scores'][:-1,1]
            elif self.scoring_method == 'SelfCheckGPT':
                raw_answer_scores = content_dict['all_hallu_scores']
            
            if unknown_filtering > 0.:
                selected_raw_pos = np.where(raw_answer_scores > unknown_filtering)[0] # if above 0.5, then it means it contradicts the greedy gold_answer
            else:
                selected_raw_pos = range(len(raw_answer_sample))
            filtered_raw_answer_sample = [(raw_answer_sample[i],raw_answer_scores[i]) for i in selected_raw_pos]
            filtered_raw_answer_sample = [r for r in filtered_raw_answer_sample if r[0].strip() != '']
            
            shortest_len = min(min(len(gold_answer_sample),len(filtered_raw_answer_sample)),4) # keep the shortest length and max at 4
            if question_filtering < 1.0 and unknown_filtering > 0.:
                gold_answer_sample = sorted(gold_answer_sample,key = lambda x:x[1],reverse=False)[:shortest_len-1]
                filtered_raw_answer_sample = sorted(filtered_raw_answer_sample,key = lambda x:x[1],reverse=True)[:shortest_len]
                filtered_raw_answer_sample = sorted(filtered_raw_answer_sample,key = lambda x:x[1],reverse=False) # sort back to smallest to widen the gap.
            else:
                gold_answer_sample = random.sample(gold_answer_sample,shortest_len-1)
                filtered_raw_answer_sample = random.sample(filtered_raw_answer_sample,shortest_len)
            if len(filtered_raw_answer_sample) == 0:
                filtered_raw_answer_sample = [random.sample(raw_answer_sample,1)[0]]
            
            out_dict = [{
                'instruction':instruction,
                'topic':topic,
                'category': category,
                'chosen_ans':chosen_ans,
                'rejected_ans':filtered_raw_answer_sample[0][0],
                'question_score':question_score
                }]
            if len(filtered_raw_answer_sample) <= 1 or len(gold_answer_sample) == 0: # only 1 sample,
                return out_dict
            
            for g,r in zip(gold_answer_sample,filtered_raw_answer_sample[1:]):
                out_dict.append({
                'instruction':instruction,
                'topic':topic,
                'category': category,
                'chosen_ans':g[0],
                'rejected_ans':r[0],
                'question_score':question_score
                })
        else:
            out_dict = {
                    'instruction':instruction,
                    'topic':topic,
                    'category': category,
                    'chosen_ans':chosen_ans,
                    'rejected_ans':rejected_ans,
                    'question_score':question_score,
                    'document':content_dict['document'],
                    'rejected_score':rejected_hallu_score if self.scoring_method == 'SelfCheckGPT' else None
                    }
            
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
        

class LLMJudge(): # use GPT3.5/4 to judge 2 given responses
    def __init__(self,engine = 'gpt-3.5-turbo-0125'):
        self.client = OpenAI()
        self.eval_prompt = [{'role':'system','content':'You are an unbiased judge, who evaluates and rank large language models (LLMs) based on the quality of their responses to the given question.'},
                            {'role':'user','content':"You are given a question, a document and two responses. Your role is to decide which is the better response.\nYou are to make your judgement based on the truthfulness of the responses with respect to the document, and also how well the response addresses the question.\nYou must respond with only either A or B.\n\nQuestion: {question}\n\nDocument: {document}\n\nResponse A: {response_a}\n\nResponse B: {response_b}"}]
        self.eval_prompt_wo_document = [{'role':'system','content':'You are a highly efficient assistant, who evaluates and rank large language models (LLMs) based on the quality of their responses to given questions.'},
                            {'role':'user','content':"You are given a question and two responses. Your role is to decide which is the better response.\nYou are to make your judgement based on the truthfulness of the response and also how well the response addresses the question.\nYou must respond with only either A or B.\n\nQuestion: {question}\n\nResponse A: {response_a}\n\nResponse B: {response_b}"}]
        
        self.engine = engine
        self.max_concurrent_calls = 10
        self.max_response_tokens = 3
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        self.cache_path = f'llm_judge_cache/{engine}.pkl'
        os.makedirs(os.path.dirname(self.cache_path),exist_ok=True)
        self.load_cache()
    
    def get_openai_choice(self,messages):
        message = messages['message']
        correct_order = messages['correct_order']
        wrong_order = messages['wrong_order']
        instr = messages['instruction']
        total_in,total_out = 0,0
        pair_result = []
        for i in range(2):
            num_tries = 0
            while num_tries < 3:
                response = None
                try:
                    response = self.client.chat.completions.create(
                            model=self.engine,
                            messages=message[i],
                            temperature=0.3,
                            max_tokens=self.max_response_tokens,
                            )
                except Exception as e:
                    time.sleep(5)
                    print (e)
                num_tries += 1  
                if response is not None:
                    resp =  response.choices[0].message.content
                    inp_tokens = response.usage.prompt_tokens
                    out_tokens = response.usage.completion_tokens
                    total_in += inp_tokens
                    total_out += out_tokens
                    if resp.lower() != 'a' and resp.lower() != 'b':
                        split_resp = resp.split()
                        split_resp = [r.translate(self.punctuation_translator) for r in split_resp] # remove punctuation
                        pred = [r for r in split_resp if len(r) == 1 and (r== 'A' or r == 'B')]
                        if len(pred) != 1: # either no match or more than 1 answer
                            pred = None
                        else:
                            pred = pred[0].lower()
                    else:
                        pred = resp.lower()
                    if pred is not None and pred in ['a','b']:
                        pair_result.append(pred)
                        break
        if len(pair_result) == 2:
            if pair_result == correct_order:
                result = 'win'
            elif pair_result == wrong_order:
                result = 'lose'
            else:
                result = 'tie'
            return {'result':result,
                    'instr': instr,
                    'in':total_in,
                    'out':total_out}
        else:
            return None
    
    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path,'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def save_cache(self):
        with open(self.cache_path,'wb') as f:
            pickle.dump(self.cache,f)
        
        
    def evaluate(self,content_dicts,track=False):
        all_messages = []
        results = []
        for content_dict in content_dicts:
            instruction = content_dict['instruction']
            base_response = content_dict['base_response']
            post_response = content_dict['post_response']
            document = content_dict['document']
            correct_order = ['a','b']
            wrong_order = ['b','a']
            eval_prompt = []
            for i in range(2):
                # if 'gpt-3.5' in self.engine:
                curr_content = deepcopy(self.eval_prompt)
                # else:
                #     curr_content = deepcopy(self.eval_prompt_wo_document)
                if i == 0:
                    curr_content[1]['content'] = curr_content[1]['content'].format(question = instruction,response_a = post_response,response_b = base_response,document = document)
                else:
                    curr_content[1]['content'] = curr_content[1]['content'].format(question = instruction,response_a = base_response,response_b = post_response,document = document)
                eval_prompt.append(curr_content)
            
            if eval_prompt[0][-1]['content'] not in self.cache:
                all_messages.append({'message':eval_prompt,'correct_order':correct_order,'wrong_order':wrong_order,'instruction':instruction})
            else:
                results.append(self.cache[eval_prompt[0][-1]['content']])

        
        remaining_results = async_process(self.get_openai_choice,all_messages,self.max_concurrent_calls)
        
        len_no_counts = [1 if r is None else 0 for r in remaining_results]
        non_empty_idx = [i for i,r in enumerate(remaining_results) if r is not None]
        remaining_results = [remaining_results[i] for i in non_empty_idx]
        out_results = [r['result'] for r in remaining_results]
        all_messages = [all_messages[i] for i in non_empty_idx]
        
        # Cache it
        for message,rel in zip(all_messages,out_results):
            self.cache[message['message'][0][-1]['content']] = rel

        self.save_cache()
        
        total_in_tokens = sum([r['in'] for r in remaining_results])
        total_out_tokens = sum([r['out'] for r in remaining_results])
        
        results.extend(out_results)
        
        # compute cost
        if 'gpt-4' in self.engine:# use gpt-4o
            cost = 0.005 * (total_in_tokens/1000) + 0.015 * (total_out_tokens/1000)
        else:
            cost = 0.0005 * (total_in_tokens/1000) + 0.0015 * (total_out_tokens/1000)

        return results,round(cost,3),sum(len_no_counts)
        
        
            
        
        
                
        
        
        
            
      
                            
            
            
            