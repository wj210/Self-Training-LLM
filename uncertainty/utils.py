from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
import re
import concurrent.futures
import os
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict
from scipy.spatial import KDTree
from accelerate import Accelerator
import random
from tqdm import tqdm
from openai import OpenAI

SCORE_KEY = {'semantic_consistency':'entropy','BSDetector':'confidence','SelfCheckGPT':'hallucination'}

def load_hf_model(model_name,quantized=False): ## if use tgi, dont quantize.
    if quantized:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None
    
    base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map= "cuda" if torch.cuda.is_available() else "cpu"
    )
    return base_model

def load_tokenizer(model_name,padding_side = "",truncation_side = ""):
    tokenizer =  AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,)
    if padding_side != "":
        tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if truncation_side != "":
        tokenizer.truncation_side = truncation_side
    tokenizer.model_max_length = 4096
    if 'mistral-7b' in model_name.lower():
        if 'instruct' not in model_name.lower():
            chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            tokenizer.chat_template = chat_template
    elif 'llama' in model_name.lower():
        if 'chat' not in model_name.lower():
            chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            tokenizer.chat_template = chat_template
    elif 'zephyr' in model_name.lower():
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    return tokenizer

def openai_call(model,message,max_tokens,temperature=0.):
    client = OpenAI()
    max_calls = 5
    num_calls = 0
    while True:
        if num_calls >= max_calls:
            return None,None
        try:
            prompt = ''
            for m in message:
                prompt += m['content']
                if m['role'] == 'assistant':
                    prompt += '\n\n'
            if 'instruct' in model.lower():
                response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                )
                cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
                return response.choices[0].text,cost
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
                cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
                return response.choices[0].message.content,cost
        except Exception as e:
            num_calls += 1
            print(f'Failing Openai call due to {e}, remaining calls: {max_calls - num_calls}')

def cal_cost(model_name,in_tokens,out_tokens):
    if model_name == 'gpt-4-0125-preview':
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

def HF_generate(inps,model,tokenizer,gen_kwargs,use_tgi=False,return_probs=False,max_workers= 64,return_as_dict=True,dict_keys={},msg=''):
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
            tokenized_inps = tokenizer(inp_b, padding='longest', return_tensors="pt",truncation=False).to(model.device)
            if return_probs:
                gen_kwargs = {'return_dict_in_generate':True,'output_scores':True, **gen_kwargs}
            with torch.no_grad():
                model_outputs = model.generate(**tokenized_inps, **gen_kwargs)
            decoded = tokenizer.batch_decode(model_outputs.sequences[:,tokenized_inps.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
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

def batch_ops(fn,inputs,batch_size,msg =''):
    out = []
    for batch_i in tqdm(range(0,len(inputs),batch_size),total = len(inputs)//batch_size,desc=msg):
        batch = inputs[batch_i:batch_i+batch_size]
        out_batch = fn(batch)
        out.extend(out_batch)
    return out

def get_extract_template(model_name,instruction_tuned): # get extract template (only used if TGI not used)
    if instruction_tuned:
        if 'neural-chat-7b' in model_name:
            extract_fn = lambda x: x.split("### Assistant:\n")[-1].strip()
        elif 'mistral' in model_name.lower() or 'zephyr' in model_name.lower():
            extract_fn = lambda x : x.split(" [/INST]")[-1].strip()
        elif 'llama' in model_name.lower():
            extract_fn = lambda x: x.split("<|assistant|>\n")[-1].strip()
        else:
            extract_fn = lambda x: x
    else:
        extract_fn = lambda x: x.split('Q:')[0].strip()
        
    return extract_fn

def format_response(response,model_name,tokenizer,mode = 'question'):
    """
    if mode = question, add system prompt and generation
    if mode = answer, add generation
    if model = label, add nothing.
    response is assumed to already be in [{'role':'user','content':str}]
    """
    if mode == 'question':
        system_prompt = "You are a student who is eager to learn about new things. You are to form a question that you lack knowledge in." # only use if question generation
    else:
        system_prompt = ''
    
    if 'mistral' not in model_name.lower() and mode != 'training_label': # Mistral doesnt have system prompt
        sys_msg = [{'role':'system','content':system_prompt}]
        response = sys_msg + response
    if 'training' not in mode:
        formatted_msg = tokenizer.apply_chat_template(response,tokenize=False,add_generation_prompt=True)
    else:
        formatted_msg = tokenizer.apply_chat_template(response,tokenize=False,add_generation_prompt=False)
    
    return formatted_msg

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
        

def format_instruction_response(model_name): # only used for SFT training
    """
    Add markers directly without chat template
    return the formatting func and response_fn
    """
    if 'mistral' in model_name.lower():
        return "<s>[INST] {instruction} [/INST] {output}</s>", "[/INST]"
    elif 'llama' in model_name.lower() or 'zephyr' in model_name.lower():
        return "<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>", "<|assistant|>"

def check_question(question):
    if 'based on' in question.lower() or 'according to' in question.lower():
        if 'document' in question or 'information' in question or 'text' in question or 'passage' in question:
            return False
    return True

def clean_question(question):
    if '?' in question:
        question = question.split('?')[0].strip()+ '?'
    return question

def clean_non_instructed_answer(answer):
    if 'Q:' in answer: # Clean off excess text
        answer = answer.split('Q:')[0].strip()
    else:
        answer.split('\n\n')[0].strip()
    return answer



def check_answer(ds,key = 'ref_answer',use_tgi=False):
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
        if 'there is no' in ans_text:
            if 'mention' in ans_text or 'record' in ans_text or 'information' in ans_text or 'data' in ans_text:
                continue
        if 'i apologize' in ans_text or 'i cannot answer' in ans_text:
            continue
        if 'Question:' in ans_text:
            ans_text = ans_text.split('Question:')[0].strip()
        if not not_str:
            d[key] = ans_text
        else:
            if use_tgi:
                ans.generated_text = ans_text
            else:
                ans['text'] = ans_text
            d[key] = ans
        filtered_ds.append(d)
    return filtered_ds




def filter_none(x):
    return [xx for xx in x if xx is not None]

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

def if_instruction_tuned(model_name):
    if 'mistral-7b' in model_name.lower():
        if 'instruct' in model_name.lower():
            return True
        else:
            return False
    elif 'llama' in model_name.lower():
        if 'chat' in model_name.lower():
            return True
        else:
            return False
    elif 'mixtral-8x7b' in model_name.lower():
        return True
    elif 'zephyr' in model_name.lower():
        return True
    else:
        raise NotImplementedError

def return_question_type(data,scoring_method,type_ = 'unknown'):
    if scoring_method == 'semantic_consistency':
        unknown_qns = [x for x in data if len(list(x['semantic_clusters'].keys())) >1] # remove questions with only 1 cluster since we cant select rejected/chosen from 1 cluster.
        known_qns = [x for x in data if len(list(x['semantic_clusters'].keys())) == 1]
    elif scoring_method == 'BSDetector':
        unknown_qns = [x for x in data if x['confidence'] < 0.5] # remove questions with confidence > 0.5
        known_qns = [x for x in data if x['confidence'] >= 0.5]
    elif scoring_method == 'SelfCheckGPT':
        unknown_qns = [x for x in data if x['hallucination'] >= 0.5] # remove questions with confidence > 0.5
        known_qns = [x for x in data if x['hallucination'] < 0.5]
    if type_ == 'unknown':
        return unknown_qns
    return known_qns

def process_document(args):
    d, all_topics = args
    if d['title'] in all_topics:
        if len(d['text'].split()) > 200:
            return d['title'], d['text']
        else:
            return None, None
    return None, None

def join_non_instruct(messages):
    out = ''
    for m in messages:
        out += m['content']
        if m['role'] == 'assistant':
            out += '\n\n'
    return out