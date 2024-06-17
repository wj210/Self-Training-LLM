from datasets import load_dataset,get_dataset_config_names,load_metric
from datasets import Dataset as HFDataset
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from multiprocessing import Pool
import copy
from utils import format_response,return_question_type
from templates import QA_format,format_answer,format_message,question_generation_examples
from topic_generator import get_predefined_topics
from typing import Union
import json
import pickle
from utils import SCORE_KEY
import re


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

def compute_rouge(scorer,h,r):
    score = scorer.score(h,r)
    return score['rougeL'].fmeasure

def is_duplicate(scorer,h,r_list,max_rouge=0.7):
    max_worker = min(64,len(r_list))
    
    with Pool(max_worker) as p:
        compute_list = [(scorer,h,r) for r in r_list]
        out = p.starmap(compute_rouge,compute_list)
    return np.max(out) > max_rouge

def chunk_document(document,tokenizer,sentence_processor,max_length,max_batches=7,return_leftover=False):
    document_sentences = [s.text for s in sentence_processor(document).sents]
    header = document_sentences[0] # take 1st sentence as header.
    document = join_sentences(document_sentences[1:])
    header_length = len(tokenizer.encode(header))
    sections = section_document(document)
    if len(sections) == 0:
        return None
    list_of_chunks = []
    taken_sections = 0
    previous_chunks = []
    current_chunk= []
    previous_lengths = 0
    for section in sections:
        taken_sections += 1
        sentences = [s.text for s in sentence_processor(section).sents]
        lengths = [len(tokenizer.encode(s)) for s in sentences]
        if sum(lengths) + header_length + previous_lengths < max_length: # too short, go to next section.
            previous_chunks.extend(sentences)
            previous_lengths += sum(lengths)
            continue
        current_chunk = [header] + previous_chunks
        current_length = header_length + previous_lengths
        sentence_added = False
        for i,(sentence, length) in enumerate(zip(sentences, lengths)):
            if length > max_length:
                continue
            sentence_added = True
            # Include sentence if it fits in the current batch, else start a new batch
            if current_length + length <= max_length:
                current_chunk.append(sentence)
                current_length += length
            else:
                # Join the current batch of sentences and start a new batch
                list_of_chunks.append(join_sentences(current_chunk))
                current_chunk = [header,sentence] # always include the first sentence
                current_length = header_length + length
                if sum(lengths[i:]) < max_length: # if remaining sections are too short, we ignore.
                    break
                if len(list_of_chunks) >= max_batches:
                    break
        if sentence_added: # if already added sentence, we reset the previous chunks and length
            previous_chunks = []
            previous_lengths = 0
  
        if len(list_of_chunks) >= max_batches:
            break
                
    if len(list_of_chunks) < max_batches:
        remaining_sentence = join_sentences(current_chunk)
        if len(tokenizer.encode(remaining_sentence)) > max_length *3/4:
            list_of_chunks.append(remaining_sentence)
    if len(list_of_chunks) == 1 and list_of_chunks[0] == "": # if somehow the document fail to be split and is too big.
        return None
    if return_leftover:
        return join_sentences([header] + sections[taken_sections:]) # return the behind portion of the document after max_length
    return list_of_chunks[:max_batches]

def section_document(document):
    sections = []
    curr_pos = 0
    splitted_document = document.split('\n')
    for i,doc in enumerate(splitted_document):
        if i != 0 and len(doc.strip().split()) <= 10 and doc.strip() != "": # is header
            sections.append('\n\n'+'\n'.join(splitted_document[curr_pos:i]))
            curr_pos = i
    if len('\n'.join(splitted_document[curr_pos:]).split()) > 20: # if too short, we ignore it
        sections.append('\n\n'+ '\n'.join(splitted_document[curr_pos:]))
    return sections
            

def join_sentences(sentences):
    processed_sentences = [f" {sent}" if i > 0 and not sentences[i-1].endswith('\n') else sent
                       for i, sent in enumerate(sentences)]
    joined =  "".join(processed_sentences)
    joined = re.sub(r'\n{3,}', '\n\n', joined) # remove multiple new lines
    joined = re.sub(r' +', ' ', joined) # remove multiple spaces
    return joined


def categorise_test_sample(ds_name,sample,type='generation'):
    if 'mmlu' in ds_name:
        return QA_sample(instruction=sample['question'],
                        choices=sample['choices'],
                        answer=sample['answer'],
                        )
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

def load_test_ds(config,test_path='',max_test_samples=100,shuffle=False,few_shot=False):
    ds_name = config['ds_name']
    num_fs = config['test_few_shot']
    if test_path != '':
        with open(test_path,'r') as f:
            test_ds = [json.loads(l) for l in f.readlines()]
        if shuffle:
            random.shuffle(test_ds)
        if max_test_samples > 0:
            test_ds = test_ds[:max_test_samples]
        if few_shot:
            out_fs = {}
            for cat,shots in question_generation_examples.items():
                out_fs[cat] = [{'instruction':s['instruction'],'answer':s['answer']} for s in shots]
        else:
            out_fs = []
        return test_ds,out_fs
    if 'mmlu' in ds_name: # get fs from each topic
        ds = load_dataset(ds_name,config['subset'],split = config['test_split'])
        val_ds = load_dataset(ds_name,config['subset'],split = 'validation').shuffle(seed=42).select(range(num_fs))
        out_fs = []
        test_ds = []
        for d in ds:
            test_ds.append(vars(categorise_test_sample(ds_name,d)))
        for vd in val_ds:
            out_fs.append(vars(categorise_test_sample(ds_name,vd)))
    elif 'truthful_qa' in ds_name:
        ds = load_dataset(ds_name,config['subset'],split = config['test_split'])
        test_ds = [vars(categorise_test_sample(ds_name,d,type=config['subset'])) for d in ds]
        if config['subset'] =='generation':
            out_fs = num_fs # fs is already defined file.
        else:
            out_fs = []
            for d in test_ds:
                d['instruction'] = num_fs + '\n\nQ: ' + d['instruction'] + '\nA:'
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
                answer = 0
            else:
                chosen_answer = d['right_answer']
                answer = 1
            question = prefix + "\n\n#Question#: " + d['question'] +"\n#Answer#: " + chosen_answer + "\n#Your Judgement#: "
            new_d.append({'instruction':question,
                          'answer':answer,
                          'system_prompt':"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be Yes or No",
                          'choices':['Yes','No'],
                          'topic':'qa'})
        test_ds = new_d
        out_fs = []
    else:
        raise ValueError(f'Unsupported dataset {ds_name}')
    return test_ds,out_fs

def load_eval_ds():
    ds = load_dataset('cais/mmlu','all',split = 'validation')
    out_ds = []
    for d in ds:
        chosen = d['choices'][d['answer']]
        remaining = [c for c in d['choices'] if c != chosen]
        rejected = random.choice(remaining)
        out_ds.append({'instruction':d['question'],
                       'chosen_ans':chosen,
                       'rejected_ans':rejected})
    return out_ds

def load_train_ds(ds_name,tokenizer,existing_ds=None):
    if existing_ds is not None: # load pre-existing self-curated SFT dataset
        out_ds = [format_message({'instruction':d['instruction'],'answer':d['answer']},tokenizer,base=False) for d in existing_ds]
        return {'train':HFDataset.from_dict({'text':out_ds})}
    else:
        
        if ds_name == 'mmlu':
            ds = load_dataset('cais/mmlu','all',split = 'auxiliary_train')
            out = []
            for d in ds:
                out.append({'instruction':d['question'],
                            'answer':d['choices'][d['answer']]})
            return {'train':train_ds}
        elif ds_name == 'ultrachat':
            ds = load_dataset("HuggingFaceH4/ultrachat_200k")
            train_ds = ds['train_sft']
            val_ds = ds['test_sft']
            column_names = list(train_ds.features)
            if tokenizer.chat_template is None:
                raise ValueError('Chat template is not defined for tokenizer')
            
            def map_fn(example,tokenizer):
                message = example['messages']
                example['text'] = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=False)
                return example
            
            train_ds = train_ds.map(map_fn,fn_kwargs={'tokenizer':tokenizer},remove_columns=column_names,num_proc=64,desc = 'applying chat template')
            val_ds = val_ds.map(map_fn,fn_kwargs={'tokenizer':tokenizer},remove_columns=column_names,num_proc=64,desc = 'applying chat template')
            return {'train':train_ds,'val':val_ds}
        elif ds_name == 'ultrafeedback':
            ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
            total = 30000
            train_ds = ds['train_prefs'].shuffle(seed=42).select(range(total))
            val_ds  = ds['test_prefs']
            def map_fn(ds):
                out_ds = []
                for d in ds:
                    instr = d['chosen'][0]['content']
                    chosen = d['chosen'][-1]['content']
                    rejected = d['rejected'][-1]['content'] 
                    out_d = {'instruction':instr,
                            'chosen_ans':chosen,
                            'rejected_ans':rejected}
                    out_ds.append(out_d)
                return out_ds
            train_ds = map_fn(train_ds)
            val_ds = map_fn(val_ds)
            return {'train':train_ds,'val':val_ds}
        elif ds_name == 'alpaca':
            ds = load_dataset("yahma/alpaca-cleaned")
            train_ds = ds['train']
            out_ds = []
            for d in train_ds:
                out_ds.append({'instruction':d['instruction'],
                            'answer': d['output'],
                            'input':d['input']})
            return {'train':out_ds}
        else:
            raise ValueError(f'Unsupported dataset {ds_name}')

def load_squad_ds(existing_ds,tokenizer):
    exisitng_ds_size = len(existing_ds)
    size_to_take = exisitng_ds_size//2
    squad_ds = load_dataset("rajpurkar/squad",split = 'train')
    squad_titles = list(set([d['title'] for d in squad_ds]))
    squad_titles = set([" ".join(t.split('_')) for t in squad_titles])
    existing_titles = set([d['topic'] for d in existing_ds])
    overlapping_titles = squad_titles.intersection(existing_titles)
    squad_ds = [d for d in squad_ds if " ".join(d['title'].split('_')) not in overlapping_titles]
    squad_context_lengths = [len(tokenizer.encode(d['context'])) for d in squad_ds]
    squad_ds = [d for d,l in zip(squad_ds,squad_context_lengths) if l <= 384 and l > 200] # take the lengths that are around our context length.
    
    squad_answer_lengths = [len(tokenizer.encode(d['answers']['text'][0])) for d in squad_ds]
    
    higher_len_samples = [d for d,l in zip(squad_ds,squad_answer_lengths) if l > 20] # first take the longer lengths
    if len(higher_len_samples) < size_to_take:
        lower_len_samples = [d for d,l in zip(squad_ds,squad_answer_lengths) if l <= 20]
        higher_len_samples.extend(random.sample(lower_len_samples,size_to_take-len(higher_len_samples)))
    
    out_ds = []
    for d in higher_len_samples:
        out_ds.append(format_message(
            {
            'instruction':format_answer(d['question'],document=d['context']),
            'answer':d['answers']['text'][0]
            },
            tokenizer,base=False))
    return HFDataset.from_dict({'text':out_ds})


def get_wiki(num_topics,data_config,get_ds=False,num_test_topics=0,topic2docu=None,existing_test_topics=None):
    if get_ds:
        ds = load_dataset(data_config['ds_name'], data_config['subset'],split='train', num_proc=16)
        return ds
    # selected_topics,topic2docu,test_topics = get_top_articles(num_topics,topic2docu,num_test_topics)
    selected_topics,topic2docu,test_topics = get_predefined_topics(num_topics,topic2docu,num_test_topics,existing_test_topics)
    return selected_topics,topic2docu,test_topics

def map_list_of_dicts(source,target,compare_key,map_key):
    """
    given 2 list of dicts, insert the map key values from source to target
    index using the compare key
    """
    source_map = {s[compare_key]:s[map_key] for s in source}
    new_target = []
    for t in target:
        t_key_value = t[compare_key]
        if t_key_value in source_map:
            t[map_key] = source_map[t_key_value]
            new_target.append(t)
    return new_target

def normalize_ds_seq_length(ds,tokenizer):
    out_ds = []
    mean_diff = []
    chosen_len = [len(tokenizer.encode(d['chosen_ans'])) for d in ds]
    rejected_len = [len(tokenizer.encode(d['rejected_ans'])) for d in ds]
    diff = [abs(c-r) for c,r in zip(chosen_len,rejected_len)]
    mean_diff = np.mean(diff)
    rejected_mask = [1 if d > mean_diff else 0 for d in diff]
    out_ds = [d for d,m in zip(ds,rejected_mask) if m == 0]
    return out_ds
    

class LikelihoodDS(torch.utils.data.Dataset):
    def __init__(self,ds,tokenizer,model_name,few_shots=None,kwargs=None):
        self.ds = ds
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.few_shots = few_shots
        self.ds_name = kwargs.get('ds_name',None)
        self.num2alpha = {i:chr(i+97).upper() for i in range(26)}
        self.alpha2token = {chr(i+97).upper():self.tokenizer.encode(chr(i+97).upper(),add_special_tokens=False)[0] for i in range(26)}
        self.setup()
    
    def setup(self):
        self.batched_ds = []
        for d in self.ds: # d is QA_sample, we batch the choices up and then process before converting back according to id
            if 'truthful_qa' in self.ds_name:
                d = self.shuffle_choices(d)
            choices = d['choices']
            answer = d['answer'] # position of choices
            test_prompt = self.format_instruction(d,add_answer=False)
            if self.few_shots is not None and len(self.few_shots) > 0:
                few_shots = self.setup_fewshot()
                test_prompt = few_shots + [test_prompt]

            formatted_instr = format_message(test_prompt,self.tokenizer,base=False)
            # for choice_id,choice in enumerate(choices):
            #     choice_alpha = self.num2alpha[choice_id]
            #     cont = f"{choice_alpha}: {choice}" if self.ds_name not in ['halueval','truthful_qa_mc'] else choice
            #     joined_context = formatted_instr + cont
            #     joined_tokenized = torch.tensor(self.tokenizer.encode(joined_context,add_special_tokens=False),dtype=torch.long)
            #     cont_tokenized = torch.tensor(self.tokenizer.encode(cont,add_special_tokens=False),dtype=torch.long)
            #     input_tokenized = joined_tokenized[:-cont_tokenized.shape[0]]
            #     combined_tokenized = torch.cat([input_tokenized,cont_tokenized])[:-1] # remove last token
            #     data_dict = {'instruction':combined_tokenized,'answer':answer,'topic':topic,'cont':cont_tokenized,'sample_id':sample_id,'choice_id':choice_id,'input_len':combined_tokenized.shape[0],'num_choices':len(choices)}
            data_dict = {'instruction':formatted_instr,
                         'answer':self.num2alpha[answer],
                         'choices':[self.num2alpha[i] for i in range(len(choices))],
                         }
            
            self.batched_ds.append(data_dict)
        return self.batched_ds
    
    def __len__(self):
        return len(self.batched_ds)
    
    def __getitem__(self,idx):
        return self.batched_ds[idx]
    
    def collate_fn(self,batch):
        instr = [b['instruction'] for b in batch]
        answer = [b['answer'] for b in batch]
        # sample_id = [b['sample_id'] for b in batch]
        # choice_id = [b['choice_id'] for b in batch]
        # cont = [b['cont'] for b in batch]
        # input_len = [b['input_len'] for b in batch]
        # num_choices = [b['num_choices'] for b in batch]
        choices = [b['choices'] for b in batch]
        tokenized_input = self.tokenizer(instr,return_tensors='pt',padding='longest',truncation=False,add_special_tokens = True)
        out = {'input_ids':tokenized_input,
                'answer':answer,
                'choice':choices
                # 'sample_id':sample_id,
                # 'choice_id':choice_id,
                # 'cont':cont,
                # 'input_len':input_len,
                # 'num_choices':num_choices
                }
        return out
    
    # def derive_prediction(self,logits,batch,pred_dict,answer_dict,num_choices_dict):
    #     logprobs = torch.nn.functional.log_softmax(logits,dim=-1)
    #     answer = batch['answer']
    #     sample_id = batch['sample_id']
    #     choice_id = batch['choice_id']
    #     input_len = batch['input_len']
    #     conts = batch['cont']
    #     num_choices = batch['num_choices']
    #     for logprob,inp_len,cont,s_id,c_id,ans,nc in zip(logprobs,input_len,conts,sample_id,choice_id,answer,num_choices):
    #         logprob = logprob[:inp_len]
    #         cont_logprob = logprob[-cont.shape[0]:]
    #         cont_logprob = torch.gather(cont_logprob,1,cont.unsqueeze(1)).squeeze(-1)
    #         normalized_logprob = cont_logprob.sum().item()/cont_logprob.shape[0]
    #         if c_id not in pred_dict[s_id]:
    #             pred_dict[s_id][c_id] = normalized_logprob
    #         if s_id not in answer_dict:
    #             answer_dict[s_id] = ans
    #         if s_id not in num_choices_dict:
    #             num_choices_dict[s_id] = nc
    #     # return pred_dict,answer_dict
    
    def derive_prediction(self,logits,batch):
        logprobs = torch.nn.functional.softmax(logits[:,-1],dim=-1)
        answer = batch['answer']
        choices = batch['choice']
        score = []
        for i,logprob in enumerate(logprobs):
            choice_probs = {}
            for choice in choices[i]:
                choice_pos = self.alpha2token[choice]
                choice_probs[choice] = logprob[choice_pos].item()
            greedy_choice = sorted(choice_probs.items(),key=lambda x:x[1],reverse=True)[0][0]
            if greedy_choice == answer[i]:
                score.append(1)
            else:
                score.append(0)
        return np.mean(score)
        
        
    def format_instruction(self,sample,add_answer=False): # sample is a QA_sample
        if self.ds_name not in ['halueval']:
            instr = sample['instruction']
            choices = sample['choices']
            instr = QA_format(instr,choices)
        else:
            instr = sample['instruction']
        msg = {'instruction':instr}
        if add_answer:
            answer = self.num2alpha[sample['answer']]
            answer_choice = choices[sample['answer']]
            msg['answer'] = f"{answer}: {answer_choice}"
        return msg
    
    def setup_fewshot(self):
        few_shots = []
        for fs in self.few_shots:
            if 'truthful_qa' in self.ds_name:
                fs = self.shuffle_choices(fs)
            few_shots.append(self.format_instruction(fs,add_answer=True))
        return few_shots
    
    def shuffle_choices(self,sample):
        choices = sample['choices']
        answer = sample['answer']
        random_indices = list(range(len(choices)))
        random.shuffle(random_indices)
        sample['choices'] = [choices[i] for i in random_indices]
        sample['answer'] = random_indices.index(answer)
        return sample
        

class GenerationDS(torch.utils.data.Dataset):
    def __init__(self,ds,tokenizer,model_name,few_shots=None,kwargs=None,scorer=None):
        self.ds = ds
        self.tokenizer = tokenizer
        self.tokenizer.padding_size = 'left' # we want to pad on the left
        self.model_name = model_name
        self.few_shots = few_shots
        self.ds_name = kwargs.get('ds_name',None)
        self.chat = True if 'chat' in self.model_name.lower() or 'zephyr' in self.model_name.lower() else False
        
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
        self.scorer = scorer 
        if 'wiki' in self.ds_name:
            self.fs = kwargs['factscorer']
            self.knowledge_source = kwargs['knowledge_source']
            self.topic2docu  = kwargs['topic2docu']
        self.setup()
        
    def setup(self):
        self.batched_ds = []
        for d in self.ds:
            instr = []
            category = d['category']
            if len(self.few_shots) == 0:
                few_shot = []
            elif isinstance(self.few_shots,dict):
                few_shot = self.few_shots[category]
            else:
                few_shot = self.few_shots
            for fs in few_shot:
                instr.append({'instruction':fs['instruction'],
                              'answer':fs['answer']})
            instr.append({'instruction':d['instruction'] if 'wiki' in self.ds_name or 'cf' in self.ds_name else d['instruction']})
            formatted_instr = format_message(instr,self.tokenizer,base=False,chat=True)
            if self.ds_name == 'truthful_qa':
                answer = {'correct_answer':d['correct_answer'],
                          'incorrect_answer':d['incorrect_answer']}
            elif 'wiki' in self.ds_name:
                answer = d['document'] # no answer provided in wiki test set, only document
            elif self.ds_name == 'cf':
                answer = None
            else:
                raise ValueError(f'Unsupported dataset {self.ds_name}')
            topic = d['topic']
            self.batched_ds.append({'instruction':formatted_instr,'answer':answer,'topic':topic,'plain_instruction':d['instruction']})
            
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
                'topic':topics,
                'plain_instruction':[b['plain_instruction'] for b in batch]
                }
    
    def score_prediction(self,pred,answer,topic,instruction=None,extract_ans_fn = None,batch_size = 16):
        pred = extract_ans_fn(pred) if extract_ans_fn is not None else pred
        if self.ds_name == 'truthful_qa': # use bleurt
            correct = answer['correct_answer']
            incorrect = answer['incorrect_answer']
            scores_true = self.metric.compute(predictions=[pred]*len(correct),references=correct)['scores']
            scores_false = self.metric.compute(predictions=[pred]*len(incorrect),references=incorrect)['scores']
            return int(max(scores_true) > max(scores_false))
        elif 'wiki' in self.ds_name: 
            fs_score = self.fs.get_score(topic,pred,
                                         gamma=0,  # gamma is length penalty, remove it.
                                         knowledge_source = self.knowledge_source,
                                         verbose = True,
                                         n = 7, # n is the number of examples to teach gpt to decompose facts.
                                         batch_size = batch_size,
                                         k=5, # k is the number of retrieved passages to support the text.
                                         questions = instruction
                                         ) 
            return fs_score
        elif 'cf' in self.ds_name:
            return -1
        else:
            raise NotImplementedError(f'Scoring method not defined for {self.ds_name}')
            
    def score_hallucination(self,instruction,ref_answer,sample_answer): # require a greedy (ref) and list of sampled answers
        assert self.scorer is not None, 'Scorer is not defined!'
        score_dict =  self.scorer.get_score(instruction,ref_answer,sample_answer) 
        if score_dict is None:
            return None
        return score_dict[SCORE_KEY[self.scorer.scoring_method]]
        
        
        
        
        
    
        
            
                
        
        
        
        
    
