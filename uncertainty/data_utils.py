from datasets import load_dataset,get_dataset_config_names, Dataset,concatenate_datasets,load_metric
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import random
from rouge_score import rouge_scorer
from multiprocessing import Pool
import copy
from utils import format_response,if_instruction_tuned
from templates import QA_format,QA_format_answer,format_answer_w_document
from typing import Union
from factscore.factscorer import FactScorer
import os
import json
from utils import SCORE_KEY


@dataclass
class QA_sample: # where the answer choices are generated text and the model is asked to choose the best one via likelihood
    instruction: str = None
    choices: list = None
    answer: int = None
    topic: str = None
    baseline_score: float = 0.0

@dataclass
class Generation_sample:
    instruction: str = None
    answer: str = None
    correct_answer: Union[str,list] = None
    incorrect_answer: Union[str,list] = None
    topic: str = None

def exclude_samples(instruction,ds_name):
    if 'mmlu' in ds_name:
        if 'which of the following' in instruction.lower():
            return True
    return False

def get_fixed_topics(ds_name):
    if 'mmlu' in ds_name:
        topics = get_dataset_config_names('cais/mmlu')
        topics.remove('all')
    elif 'gsm8k' in ds_name:
        topics = ['Diverse grade school math word problems.']
    elif 'truthful' in ds_name:
        ds = load_dataset('truthful_qa','generation',split = 'validation')
        topics = set()
        for d in ds:
            topics.add(d['category'])
        topics = list(topics)
        del ds
    else:
        raise ValueError(f'Unsupported dataset {ds_name}')
    return topics 

def compute_rouge(scorer,h,r):
    score = scorer.score(h,r)
    return score['rougeL'].fmeasure

def is_duplicate(scorer,h,r_list,max_rouge=0.7):
    max_worker = min(64,len(r_list))
    
    with Pool(max_worker) as p:
        compute_list = [(scorer,h,r) for r in r_list]
        out = p.starmap(compute_rouge,compute_list)
    return np.max(out) > max_rouge

def categorise_test_sample(ds_name,sample,type='generation'):
    if 'mmlu' in ds_name:
        return QA_sample(instruction=sample['question'],
                        choices=sample['choices'],
                        answer=sample['answer'],
                        topic=sample['subject'])
    elif 'truthful_qa' in ds_name:
        if type == 'generation':
            return Generation_sample(instruction=sample['question'],
                                    answer=sample['best_answer'],
                                    correct_answer=sample['correct_answers'],
                                    incorrect_answer=sample['incorrect_answers'],
                                    topic=sample['category'])
        else:
            return QA_sample(instruction=sample['question'],
                            choices=sample['mc1_targets']['choices'],
                            answer=sample['mc1_targets']['labels'].index(1),
                            topic='truthful_qa_mc1')
    elif 'gsm8k' in ds_name:
        return Generation_sample(instruction=sample['question'],
                                answer=sample['answer'],
                                topic='gsm8k')
    else:
        raise ValueError(f'Unsupported dataset {ds_name} for categorisation.')

def load_test_ds(config):
    ds_name = config['ds_name']
    num_fs = config['test_few_shot']
    if 'mmlu' in ds_name: # get fs from each topic
        ds = load_dataset(ds_name,config['subset'],split = config['test_split'])
        out_fs = defaultdict(list)
        test_ds = []
        for d in ds:
            if len(out_fs[d['subject']]) >= num_fs:
                test_ds.append(vars(categorise_test_sample(ds_name,d)))
                continue
            out_fs[d['subject']].append(vars(categorise_test_sample(ds_name,d)))
    elif 'truthful_qa' in ds_name:
        ds = load_dataset(ds_name,config['subset'],split = config['test_split'])
        test_ds = [vars(categorise_test_sample(ds_name,d,type=config['subset'])) for d in ds]
        if config['subset'] =='generation':
            out_fs = num_fs # fs is already defined file.
        else:
            out_fs = test_ds[:num_fs] # fs is the first num_fs samples for multiple_choice
            test_ds = test_ds[num_fs:]
    elif 'wiki' in ds_name:
        ds_path = 'data/wiki/test.jsonl'
        with open(ds_path,'r') as f:
            test_ds = [json.loads(l) for l in f.readlines()]
        out_fs = []
    elif 'halueval' in ds_name:
        ds_path = 'data/halueval/qa_data.jsonl'
        with open(ds_path,'r') as f:
            test_ds = [json.loads(l) for l in f.readlines()]
        prefix_path = 'data/halueval/qa_evaluation_instruction.txt'
        with open(prefix_path,'r', encoding="utf-8") as f:
            prefix = f.read()
        new_d = []
        for d in test_ds:
            if random.random() > 0.5:
                chosen_answer = d["hallucinated_answer"]
                answer = 'Yes'
            else:
                chosen_answer = d['right_answer']
                answer = 'No'
            question = prefix + "\n\n#Question#: " + d['question'] +"\n#Answer#: " + chosen_answer + "\n#Your Judgement#: "
            new_d.append({'instruction':question,
                          'answer':answer,
                          'system_prompt':"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be Yes or No",
                          'topic':'qa'})
        test_ds = new_d
        out_fs = []
    else:
        raise ValueError(f'Unsupported dataset {ds_name}')
    return test_ds,out_fs

def get_fixed_ds(config,question_per_topic,test_qn_per_topic,generator_type=''): ## for training purpose.
    ds_name = config['ds_name']
    few_shot = config['few_shot']
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    out_ds = []
    test_topic_count = defaultdict(int)
    fs_ds = defaultdict(list)
    
    ## MMLU ##
    if 'mmlu' in ds_name:
        topics = get_fixed_topics(ds_name)
        test_ds = load_dataset(ds_name,config['subset'],split = config['test_split'])
        spare_ds = load_dataset(ds_name,config['subset'],split = 'validation').shuffle(seed=42) # get few-shot qns from val_ds
        remaining_samples = [] # collect remaining samples for predefined dataset usage.
        for d in test_ds:
            subject = d['subject']
            if test_qn_per_topic >0 and test_topic_count[subject] >= test_qn_per_topic:
                remaining_samples.append(d)
                continue
            out_ds.append(categorise_test_sample(ds_name,d))
            test_topic_count[subject] +=1 
        # for few_shot, derive from val ds
        if few_shot > 0:
            for d in spare_ds:
                subject = d['subject']
                instruction = d['question']
                answer = d['choices'][d['answer']]
                if len(fs_ds[subject]) >= few_shot:
                    remaining_samples.append(d)
                    continue
                if exclude_samples(instruction,ds_name): # don't include qn with choices.
                    continue
                exisiting_instructions = [o.instruction for o in fs_ds[subject]]
                if len(exisiting_instructions) > 0 and is_duplicate(r_scorer,instruction,exisiting_instructions): # don't include duplicate qn
                    remaining_samples.append(d)
                    continue
                fs_ds[subject].append(categorise_test_sample(ds_name,d))
                
        ## We check how many samples are left for each topic and topics with < question_per_topic, we add from other topic
        remaining_topic_count = defaultdict(int)
        taken_topic_instructions = {k:set([o.instruction for o in fs_ds[k]]) for k in topics}
        for d in remaining_samples:
            subject = d['subject']
            if d['question'] not in taken_topic_instructions[subject]:
                remaining_topic_count[subject] += 1
        remaining_topic_count = sorted(remaining_topic_count.items(),key=lambda x:x[1],reverse=True)
        collect_topic_count = {t:question_per_topic for t in topics}
        end_search=False
        for top,remaining in remaining_topic_count:
            required_num_topics = collect_topic_count[top]
            if remaining < required_num_topics:
                collect_topic_count[top] = remaining
                remainder = required_num_topics - remaining
                for _ in range(remainder):
                    topics_to_chose = [t for t,c in remaining_topic_count if c>collect_topic_count[t]]
                    if len(topics_to_chose)> 0:
                        random_chosen_topic_to_add = random.choice(topics_to_chose)
                        collect_topic_count[random_chosen_topic_to_add] += 1
                    else:
                        end_search=True
                        print ('Warning: Not enough questions to fill up the question_per_topic, consider lowering question_per_topic')
                        break
            if end_search:
                break
        ## For predefined ds
        defined_ds = []
        topic_count = defaultdict(int)
        for d in remaining_samples:
            subject = d['subject']
            ques = d['question']
            choices = d['choices']
            answer = d['answer']
            if topic_count[subject] >= collect_topic_count[subject]:
                continue
            if d['question'] in set([o.instruction for o in fs_ds[subject]]): # don't include qn which is already in the few shot examples
                continue
            if exclude_samples(d['question'],ds_name): # don't include qn with choices.
                continue
            if 'oracle' in generator_type: # upper bound on performance, SFT on actual training set
                ques = QA_format(ques,choices)
                answer = QA_format_answer(answer,choices)
            else:
                answer = None
            defined_ds.append({'topic':subject,
                               'instruction': ques,
                               'choices':choices,
                               'actual_label':answer
                               })
            topic_count[subject] += 1
            if all([(topic_count[d]) >=  collect_topic_count[d] for d in topics]):
                break

    ## TruthulQA ##
    elif 'truthful_qa' in ds_name:  # only for generation for now.
        test_ds = load_dataset(ds_name,'generation',split = config['test_split'])
        ## randomly select 5 samples from test_ds
        for d in test_ds:
            subject = d['category']
            test_sample = categorise_test_sample(ds_name,d)
            # if len(fs_ds[subject]) >= few_shot:
            out_ds.append(test_sample)
                # continue
            # fs_ds[subject].append(test_sample)
            
        defined_ds = out_ds
    
    ## GSM8K ##
    elif 'gsm8k' in ds_name:
        test_ds = load_dataset(ds_name,config['subset'],split = config['test_split'])
        spare_ds = load_dataset(ds_name,config['subset'],split = 'train').shuffle(seed=42)
        out_ds = [categorise_test_sample(ds_name,d) for d in test_ds]
        fs_ds = []
        for d in spare_ds:
            if len(fs_ds) >= few_shot:
                break
            fs_ds.append(categorise_test_sample(ds_name,d))
        spare_ds = spare_ds.select(range(len(fs_ds),len(spare_ds)))
        defined_ds = spare_ds.select(range(question_per_topic)) 

    return out_ds,fs_ds,defined_ds

def get_wiki(num_topics,data_config,get_ds=False):
    ds = load_dataset(data_config['ds_name'], data_config['subset'],split='train', num_proc=16)
    if get_ds:
        return ds
    sampled = ds.shuffle(seed=42).select(range(num_topics))
    topics = [d['title'] for d in sampled]
    topic2docu = {d['title']:d['text'] for d in sampled}
    return topics,topic2docu

class LikelihoodDS(torch.utils.data.Dataset):
    def __init__(self,ds,tokenizer,model_name,few_shots=None,kwargs=None):
        self.ds = ds
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num2alpha = {i:chr(i+97).upper() for i in range(26)}
        self.alpha2token = {chr(i+97).upper():self.tokenizer.encode(chr(i+97).upper(),add_special_tokens=False)[0] for i in range(26)}
        self.few_shots = few_shots
        self.trained = kwargs.get('trained',False)
        self.ds_name = kwargs.get('ds_name',None)
        self.setup()
    
    def setup(self):
        self.batched_ds = []
        for d in self.ds: # d is QA_sample, we batch the choices up and then process before converting back according to id
            topic = d['topic']
            possible_choices = [self.num2alpha[i] for i in range(len(d['choices']))] # [A,B,C ...]
            test_prompt = self.format_instruction(d,add_answer=False)
            if self.few_shots is not None:
                few_shots = self.setup_fewshot(topic)
                test_prompt = few_shots + test_prompt
            if self.trained or if_instruction_tuned(self.model_name):
                formatted_instr = format_response(test_prompt,self.model_name,self.tokenizer,mode='answer')
            else:
                formatted_instr = ''
                for msg in test_prompt:
                    if msg['role'] == 'user':
                        formatted_instr += msg['content']
                    elif msg['role'] == 'assistant':
                        formatted_instr += (msg['content'] + '\n\n')
            answer = self.num2alpha[d['answer']]
            topic = d['topic']
            data_dict = {'instruction':formatted_instr,'answer':answer,'topic':topic,'choices':possible_choices}
            self.batched_ds.append(data_dict)
        return self.batched_ds
    
    def __len__(self):
        return len(self.batched_ds)
    
    def __getitem__(self,idx):
        return self.batched_ds[idx]
    
    def collate_fn(self,batch):
        instr = [b['instruction'] for b in batch]
        choices = [b['choices'] for b in batch]
        answer = [b['answer'] for b in batch]
        topics = [b['topic'] for b in batch]
        # tokenized_input = pad_sequence([torch.tensor(self.tokenizer.encode(i,add_special_tokens = False)) for i in instr],batch_first=True,padding_value=self.tokenizer.pad_token_id)
        tokenized_input = self.tokenizer(instr,return_tensors='pt',padding='longest',truncation=False,add_special_tokens=False).input_ids
        out = {'input_ids':tokenized_input,
                'answer':answer,
                'topic':topics,
                'choices':choices
                }
        if 'data_sample' in batch[0]:
            out['data_sample'] = [b['data_sample'] for b in batch]
        
        return out
    
    def derive_prediction(self,logits,choices,temperature,num_samples):
        out = {}
        choice_probs = defaultdict()
        with torch.no_grad():
            scaled_logits = logits/temperature if temperature > 0 else logits
            probs = torch.nn.functional.softmax(scaled_logits,dim=0) # vocab size.
            # get greedy option 
            for choice in choices: 
                choice_probs[choice] = probs[self.alpha2token[choice]] 
            greedy_choice = sorted(choice_probs.items(),key=lambda x:x[1],reverse=True)[0][0]
            out['greedy'] = greedy_choice
            # Get sampled options for confidence scoring #
            sampled_options = torch.multinomial(probs,num_samples,replacement=True).tolist() # set replacement to True!
            out['sampled'] = self.tokenizer.batch_decode(sampled_options,skip_special_tokens=True)
            
        return out
    
    def format_instruction(self,sample,add_answer=False): # sample is a QA_sample
        instr = sample['instruction']
        choices = sample['choices']
        instr = QA_format(instr,choices)
        msg = [{'role':'user','content':instr}]
        if add_answer:
            answer = self.num2alpha[sample['answer']]
            msg += [{'role':'assistant','content':f' {answer}'}]
        return msg
    
    def setup_fewshot(self,topic):
        few_shots = []
        if self.ds_name == 'mmlu':
            shuffled_few_shots = copy.deepcopy(self.few_shots[topic])
        elif 'truthful_qa' in self.ds_name:
            shuffled_few_shots = copy.deepcopy(self.few_shots)
        random.shuffle(shuffled_few_shots)
        for fs in shuffled_few_shots:
            few_shots.extend(self.format_instruction(fs,add_answer=True))
        return few_shots
        

class GenerationDS(torch.utils.data.Dataset):
    def __init__(self,ds,tokenizer,model_name,few_shots=None,kwargs=None,scorer=None):
        self.ds = ds
        self.tokenizer = tokenizer
        self.tokenizer.padding_size = 'left' # we want to pad on the left
        self.model_name = model_name
        self.few_shots = few_shots
        self.ds_name = kwargs.get('ds_name',None)
        if self.ds_name == 'truthful_qa': # limit the gpu memory usage
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Restrict TensorFlow to only allocate necessary memory
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Exception handling
                    print(e)
            self.metric = load_metric('bleurt','bleurt-20')
        else:
            self.metric = None
        self.trained = kwargs.get('trained',False)
        self.scorer = scorer 
        # if self.ds_name == 'wiki':
        #     self.fs = kwargs['factscorer']
        #     self.knowledge_source = kwargs['knowledge_source']
        self.setup()
        
    def setup(self):
        self.batched_ds = []
        for d in self.ds:
            instr = []
            if self.trained or if_instruction_tuned(self.model_name):
                for fs in self.few_shots:
                    instr.extend([{'role':'user','content':fs["instruction"]},{'role':'assistant','content':fs["answer"]}])
                instr.append({'role':'user','content':d['instruction']})
                formatted_instr = format_response(instr,self.model_name,self.tokenizer,mode='answer')
            else:
                for fs in self.few_shots:
                    instr.append(f'Q: {fs["instruction"]}\nA: {fs["answer"]}')
                instr.append(f"Q: {d['instruction']}\nA: ")
                formatted_instr = '\n\n'.join(instr) 
            if self.ds_name == 'truthful_qa':
                answer = {'correct_answer':d['correct_answer'],
                          'incorrect_answer':d['incorrect_answer']}
            elif self.ds_name == 'wiki':
                answer = d['document'] # no answer provided in wiki test set, only document
            elif self.ds_name == 'halueval':
                answer = d['answer']
            else:
                raise ValueError(f'Unsupported dataset {self.ds_name}')
            topic = d['topic']
            self.batched_ds.append({'instruction':formatted_instr,'answer':answer,'topic':topic})
            
    def __len__(self):
        return len(self.batched_ds)
    
    def __getitem__(self,idx):
        return self.batched_ds[idx]
    
    def collate_fn(self,batch):
        instr = [b['instruction'] for b in batch]
        answer = [b['answer'] for b in batch]
        topics = [b['topic'] for b in batch]
        return {'input_ids':instr,
                'answer':answer,
                'topic':topics
                }
    
    def score_prediction(self,pred,answer,topic,extract_ans_fn = None):
        pred = extract_ans_fn(pred) if extract_ans_fn is not None else pred
        if self.ds_name == 'truthful_qa': # use bleurt
            correct = answer['correct_answer']
            incorrect = answer['incorrect_answer']
            scores_true = self.metric.compute(predictions=[pred]*len(correct),references=correct)['scores']
            scores_false = self.metric.compute(predictions=[pred]*len(incorrect),references=incorrect)['scores']
            return int(max(scores_true) > max(scores_false))
        elif self.ds_name == 'wiki': 
            fs_score = self.fs.get_score(topic,pred,
                                         gamma=0,  # gamma is length penalty, remove it.
                                         knowledge_source = self.knowledge_source,
                                         verbose = True,
                                         n = 2,
                                         batch_size = 8) # n is the number of examples to teach gpt to decompose facts.
            return fs_score
        elif self.ds_name == 'halueval':
            if answer.lower() in pred.lower().strip():
                return 1
            else:
                return 0
        else:
            raise NotImplementedError(f'Scoring method not defined for {self.ds_name}')
            
    def score_hallucination(self,instruction,ref_answer,sample_answer): # require a greedy (ref) and list of sampled answers
        assert self.scorer is not None, 'Scorer is not defined!'
        score_dict =  self.scorer.get_score(instruction,ref_answer,sample_answer) 
        return score_dict[SCORE_KEY[self.scorer.scoring_method]]
        
        
        
        
        
    
        
            
                
        
        
        
        
    
