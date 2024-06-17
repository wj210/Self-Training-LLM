from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch
import re
import concurrent.futures
import numpy as np
from typing import List, Dict
from accelerate import Accelerator
import random
from tqdm import tqdm
from openai import OpenAI
import time
from templates import format_answer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
SCORE_KEY = {'semantic_consistency':'entropy','BSDetector':'confidence','SelfCheckGPT':'hallucination'}

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def load_hf_model(model_name,quantized=False,is_adapter=False,use_flash=True): ## if use tgi, dont quantize.
    if quantized:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None
    
    if is_adapter:
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map ='cuda',
                    )
        base_model = peft_model.merge_and_unload(progressbar=True)
    else:
        if use_flash:
            base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map= "cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation = 'flash_attention_2',
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            ).to('cuda')
    return base_model

def load_tokenizer(model_name,padding_side = "",truncation_side = "",prompt_format = 'chat'):
    tokenizer =  AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,)
    if padding_side != "":
        tokenizer.padding_side = padding_side
    if truncation_side != "":
        tokenizer.truncation_side = truncation_side
    if tokenizer.chat_template is None:
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        tokenizer.chat_template = chat_template
    if 'mistral' in model_name.lower() or 'zephyr' in model_name.lower():
        tokenizer.model_max_length = 4096    
    return tokenizer

def resize_pad_embeddings(model,tokenizer): # only for alpaca-trained
    pad_token = "[PAD]"
    special_tokens_dict = dict(pad_token=pad_token)
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) 
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
    print(f"Resized tokenizer and embedding to {len(tokenizer)} tokens.")

def openai_call(model,message,max_tokens,temperature=0.,n=1):
    client = OpenAI()
    max_calls = 5
    num_calls = 0
    while True:
        if num_calls > max_calls:
            return None,None
        try:
            if 'instruct' in model.lower():
                prompt = ''
                for m in message:
                    prompt += m['content']
                    if m['role'] == 'assistant':
                        prompt += '\n\n'
                response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n = n,
                )
                cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
                if n > 1:
                    return [r.text for r in response.choices],cost
                else:
                    return response.choices[0].text,cost
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    )
                cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
                if n > 1:
                    return [r.message.content for r in response.choices],cost
                else:
                    return response.choices[0].message.content,cost
        except Exception as e:
            num_calls += 1
            time.sleep(num_calls**2)
            print(f'Failing Openai call due to {e}, remaining calls: {max_calls - num_calls}')

def cal_cost(model_name,in_tokens,out_tokens):
    if 'gpt-4' in model_name:
        cost = in_tokens * (10/1e6) + (out_tokens * (30/1e6))
    elif model_name == 'gpt-3.5-turbo-0125':
        cost = in_tokens * (0.5/1e6) + (out_tokens * (1.5/1e6))
    elif model_name == 'gpt-3.5-turbo-instruct':
        cost = in_tokens * (1.5/1e6) + (out_tokens * (2/1e6))
    else:
        raise NotImplementedError
    return cost

def tgi_to_gen_kwargs(gen_kwargs): # convert TGI kwargs to HF kwargs
    if 'details' in gen_kwargs:
        gen_kwargs.pop('details')
    if 'best_of' in gen_kwargs:
        gen_kwargs['num_return_sequences'] = gen_kwargs.pop('best_of')
    return gen_kwargs

def HF_generate(inps,model,tokenizer,gen_kwargs,use_tgi=False,return_probs=True,max_workers= 64,return_as_dict=True,dict_keys={},msg=''):
    """
    Takes in the entire set of inputs and batch it using standard HF generation, else async with tgi API.
    if return_probs, return as a dict for each sample if not using TGI, else logprobs can be directly accessed via the object returned by TGI.
    if return_as_dict, return as a dict with all original items, along with the new output specified by 'output' key, input is specified by 'input' else just returns the output.
    """
    if return_as_dict:
        assert dict_keys != {}, 'input_key must be provided if return_as_dict is True'
    if not use_tgi:
        gen_kwargs = tgi_to_gen_kwargs(gen_kwargs)
        out = []
        for batch_i in tqdm(range(0,len(inps),max_workers),total =len(inps)//max_workers,desc=msg):
            inp_batch = inps[batch_i:batch_i+max_workers]
            out_key = dict_keys.get('output',None)
            if return_as_dict:
                inp_key = dict_keys['input']
                inp_b = [inp[inp_key] for inp in inp_batch]
            else:
                inp_b = inp_batch
            tokenized_inps = tokenizer(inp_b, padding='longest', return_tensors="pt",truncation=False)
            tokenized_inps = {k:v.to(model.device) for k,v in tokenized_inps.items()}
            if return_probs:
                gen_kwargs = {'return_dict_in_generate':True,'output_scores':True, **gen_kwargs}
            with torch.no_grad():
                model_outputs = model.generate(**tokenized_inps, **gen_kwargs)
            decoded = tokenizer.batch_decode(model_outputs.sequences[:,tokenized_inps['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if return_probs:
                transition_scores = model.compute_transition_scores(model_outputs.sequences, model_outputs.scores, normalize_logits=True)
                logprobs = transition_scores.detach().cpu().numpy()
            else:
                logprobs = None
            if gen_kwargs.get('num_return_sequences',1) > 1:
                num_seq = gen_kwargs['num_return_sequences']
                decoded = [decoded[i:i+num_seq] for i in range(0,len(decoded),num_seq)]
                if logprobs is not None:
                    logprobs = [logprobs[i:i+num_seq] for i in range(0,logprobs.shape[0],num_seq)]
            if logprobs is not None:
                for d,lp,inp_b in zip(decoded,logprobs,inp_batch):
                    if return_as_dict:
                        out.append({out_key:{'text':d,'logprobs':lp},**{k:v for k,v in inp_b.items() if k != inp_key}})
                    else:
                        out.append({'text':d,'logprobs':lp})
            else:
                for d,inp_b in zip(decoded,inp_batch):
                    if return_as_dict:
                        out.append({out_key:d,**{k:v for k,v in inp_b.items() if k != inp_key}})
                    else:
                        out.append(d)
        return out
    else:
        return tgi_generate(inps,model,gen_kwargs,max_workers,return_as_dict=return_as_dict,dict_keys=dict_keys,msg=msg)

def tgi_generate(inps,model,gen_kwargs,max_workers,return_as_dict=True,dict_keys={},msg = ''): # Using TGI to generate
    def tgi_generate(inputs):
        if return_as_dict:
            prompt = inputs[dict_keys['input']]
            gen = model.text_generation(prompt, **gen_kwargs)
            return {dict_keys['output']:gen,**{k:v for k,v in inputs.items() if k != dict_keys['input']}}
        else:
            return model.text_generation(inputs, **gen_kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(inps),max_workers)) as executor:
        if msg != '':
            out = list(tqdm(executor.map(tgi_generate,inps),total = len(inps),desc = msg))
        else:
            out = list(executor.map(tgi_generate,inps))
    return out

def get_tgi_text(x):
    if x.details.best_of_sequences is not None: # means more than 1 sequence
        return [x.generated_text] + [s.generated_text for s in x.details.best_of_sequences]
    else:
        return x.generated_text

def batch_ops(fn,inputs,batch_size,msg =''):
    out = []
    for batch_i in tqdm(range(0,len(inputs),batch_size),total = len(inputs)//batch_size,desc=msg):
        batch = inputs[batch_i:batch_i+batch_size]
        out_batch = fn(batch)
        out.extend(out_batch)
    return out

def format_response(response,tokenizer):
    prompt = ""
    if not isinstance(response,list):
        response = [response]
    for r in response: # expected to be a dict
        last_response = 'instruction'
        if prompt_formatting:
            prompt += format_answer(r['instruction'],r['topic'],r.get('document','')) # DPO doesnt have document
        else:
            prompt += r['instruction']
        if r.get('answer','') != '':
            prompt += f"{r['answer']}{tokenizer.eos_token}" + '\n\n'
            last_response = 'answer'
    if last_response == 'answer':
        prompt = prompt.strip()
    return prompt

def format_fs_qa(few_shot,instruct_tuned=False):
    all_fs =[]
    random.shuffle(few_shot)
    for fs in few_shot:
        q = fs.instruction
        if hasattr(fs,'choices'):
            a = fs.choices[fs.answer] # answer is index of choice
        else:
            a = fs.answer # else is just answer 
        if instruct_tuned:
            all_fs.extend([{'role':'user','content':q},
                            {'role':'assistant','content':a}])
        else:
            all_fs.append(f"Q: {q}\nA: {a}")
    return all_fs
        

def format_instruction_response(ds_name): # only used for SFT training
    """
    Add markers directly without chat template
    return the formatting func and response_fn
    """
    if 'chat' in ds_name:
        return "<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>", "<|user|>","<|assistant|>\n"
    elif ds_name == 'alpaca':
        return None, None,"### Response:"
    else:
        raise NotImplementedError

def check_question(question,topic,check_topic=False):
    if question.strip() == "" or 'document' in question or 'text' in question or 'passage' in question or 'information' in question:
        return False
    if check_topic:
        topic_check_words = [t for t in topic.lower().split() if t not in ['the','a','an','of','in','and','on','for','from','painting']]
        if not any([t in question.lower() for t in topic_check_words]): # if any topic words not mentioned in question, it is most likely ambiguous
            return False
    return True

def clean_question(question,base=False):
    if base:
        question = clean_base_response(question,True)
    if '\n' in question:
        question = question.split('\n')[0].strip()
    if '?' in question:
        question = question.split('?')[0].strip()+ '?'
    if 'Question:' in question:
        question = question.split('Question:')[1].strip()
    
    return question

def clean_base_response(x,is_question=False):
    if '### Instruction:' in x:
        x = x.split('### Instruction:')[0].strip()
    elif '###' in x:
        x = x.split('###')[0].strip()
    # else:
    #     x = x.split('\n\n')[0].strip()
    if is_question:
        pattern = r"^\d+\.\s*(.*)" # clean any {digit}. {question} -> question
        match = re.match(pattern, x)
        if match:
            x = match.group(1)
    return x

def check_answer(ds,sent_processor,tokenizer,key = 'ref_answer',use_tgi=False,base=False):
    filtered_ds = []
    for d in ds:
        ans = d[key]
        if not isinstance(ans,str):
            not_str = True
            if use_tgi:
                ans_text = ans.generated_text
            else:
                ans_text = ans['text']
        else:
            not_str = False
            ans_text = ans
        if base:
            ans_text = clean_base_response(ans_text)
        else:
            ans_text = filter_nonhelpful_ans(ans_text,sent_processor,tokenizer)
        if ans_text == None:
            continue
        if not not_str: # return in text form if base
            d[key] = ans_text
        else:
            if use_tgi:
                ans.generated_text = ans_text
            else:
                ans['text'] = ans_text
            d[key] = ans
        filtered_ds.append(d)
    return filtered_ds

def check_single_answer(ans,sent_processor,tokenizer,base=False,max_length = 256):
    ans_text = ans.generated_text
    if base:
        ans_text = clean_base_response(ans_text)
        if len(tokenizer.encode(ans_text,add_special_tokens = False)) > (max_length -10): # base model have the tendency to get overly verbose (increase hallu)
            return None
        if check_for_duplicates(ans_text,sent_processor): # used for base model.
            return None
    ans_text  = clean_incomplete_response(ans_text,sent_processor,tokenizer,max_length = max_length)
    if ans_text == None:
        return None
    if ans_text.strip() == '':
        return None
    if not base:
        ans.generated_text = ans_text
        return ans
    else:
        return ans_text

def filter_nonhelpful_ans(ans,sent_processor,tokenizer,max_length=384):
    """
    1) Check for unhelpful answers and remove them.
    2) Check for any sentences mentioning the document and remove them.
    3) Truncate non-ending sentences and check for repetitive answers
    """
    # 1)
    sents = [s.text.strip() for s in sent_processor(ans).sents]
    initial_existence = [True if 'not' in s else False for s in sents]
    if sum(initial_existence) > 0:
        check_sentences = [s for s,i in zip(sents,initial_existence) if i]
        if any(['mentioned' in s or 'specified' in s or 'provided' in s for s in check_sentences]):
            return None
    
    if any(['document' in s for s in sents]):
        # 2)
        sents = [s for s in sents if 'document' not in s]
        if len(sents) == 0:
            return None
        # 3)
        ans = clean_incomplete_response(ans,sent_processor,tokenizer,max_length,sentences = sents,is_sentence=True)
    return ans

def filter_none(x):
    return [xx for xx in x if xx is not None]

def clean_incomplete_response(answer,sent_processor,tokenizer,max_length=384,sentences=None,is_sentence=False): # prevent incomplete sentences
    if not is_sentence or sentences is None:
        sentences = [s.text.strip() for s in sent_processor(answer).sents]
    if len(sentences) > 1 and not sentences[-1].endswith('.') and not sentences[-1].endswith('"') and len(tokenizer.encode(answer)) > max_length-50:
        sentences = sentences[:-1]
    unique_sents = list(set(sentences))
    if len(unique_sents) < len(sentences)-1 : # if 2 or more sentences are repeated, remove the answer
        return None
    return "".join([f" {sent}" if i > 0 and not sentences[i-1].endswith('\n') else sent
                    for i, sent in enumerate(sentences)])


def check_for_duplicates(text,sent_processor):
    def split_into_words(sentences):
        split_sentences = [set(re.split(r'\W+', sentence.strip())) for sentence in sentences if sentence.strip()]
        for sentence in split_sentences:
            sentence.difference_update(stop_words)
        return split_sentences
    
    def jaccard_similarity(pair):
        set1, set2 = pair
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        iou = intersection / union if union != 0 else 0
        if len(set1) > 5 or len(set2) > 5:
            threshold = 0.7
        else:
            threshold = 0.5
        return iou >= threshold
    
    if '\n' in text:
        sentences = text.split('\n')
    else:
        sentences = [sent.text.strip() for sent in sent_processor(text).sents]
    
    word_sets = split_into_words(sentences)
    if len(word_sets) <= 1:
        return False
    last_sets = max(len(word_sets)//5,1) # take last 20% of sentences
    starting_set = len(word_sets) - last_sets
    duplicates = []
    for i in range(starting_set,len(word_sets)):
        curr = [jaccard_similarity((word_sets[i], word_set)) for j,word_set in enumerate(word_sets) if j!= i]
        duplicates.append(sum(curr) > 0)
    return sum(duplicates) > 0



def extract_str_in_bracket(x):
    pattern = r"\((.*?)\)"
    match = re.search(pattern, x)
    if match:
        extracted_text = match.group(1) 
        return extracted_text
    else:
        return x

def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None

def return_question_type(data,scoring_method,type_ = 'unknown',score_threshold=0.5,scoring_key = None):
    assert type_ in ['unknown','known'], 'type_ must be either unknown or known'
    if scoring_key == None:
        scoring_key = SCORE_KEY[scoring_method]
    if scoring_method == 'semantic_consistency':
        unknown_qns = [x for x in data if len(list(x[scoring_key].keys())) >1] # remove questions with only 1 cluster since we cant select rejected/chosen from 1 cluster.
        known_qns = [x for x in data if len(list(x[scoring_key].keys())) == 1]
    elif scoring_method == 'BSDetector':
        unknown_qns = [x for x in data if x[scoring_key] < score_threshold] # remove questions with confidence > 0.5
        known_qns = [x for x in data if x[scoring_key] >= score_threshold]
    elif scoring_method == 'SelfCheckGPT':
        # unknown_qns = [x for x in data if x[scoring_key] > score_threshold or np.max(x['all_hallu_scores']) > score_threshold] # remove questions with confidence > 0.5
        unknown_qns = [x for x in data if x[scoring_key] > score_threshold ]
        known_qns = [x for x in data if x[scoring_key] <= score_threshold]
    if type_ == 'unknown':
        return unknown_qns
    return known_qns

def refine_question(x):
    check_words = ['what','why','how','when','who','where','which']
    if 'and' in x:
        x_split = x.split('and')
        starting = x_split[0].strip()
        if '.' in starting or ',' in starting or '?' in starting:
            starting = starting[:-1]
        out = [starting]
        for remaining in x_split[1:]:
            if any([c in remaining for c in check_words]):
                break
            else:
                out.append(remaining.strip())
        out = ' and '.join(out)
        if not out.endswith('?'):
            out += '?'
    else:
        out = x
    return out
