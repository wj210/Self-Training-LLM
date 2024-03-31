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

def load_tokenizer(model_name,padding_side = "left"):
    tokenizer =  AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side = padding_side)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 2048
    if 'mistral-7b' in model_name.lower():
        if 'instruct' not in model_name.lower():
            chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            tokenizer.chat_template = chat_template
    elif 'llama' in model_name.lower():
        if 'chat' not in model_name.lower():
            chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            tokenizer.chat_template = chat_template
    return tokenizer

def openai_call(model,message,max_tokens,temperature=0.):
    client = OpenAI()
    max_calls = 5
    num_calls = 0
    while True:
        if num_calls >= max_calls:
            return None
        try:
            prompt = [m['content'] for m in message if m['role'] == 'user'][0]
            if 'instruct' in model.lower():
                response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                )
                return response.choices[0].text
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
                return response.choices[0].message.content 
        except Exception as e:
            num_calls += 1
            print(f'Failing Openai call due to {e}, remaining calls: {max_calls - num_calls}')
    

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
                decoded = [d[i:i+num_seq] for i in range(0,len(d),num_seq)]
                if logprobs is not None:
                    logprobs = [lp[i:i+num_seq] for i in range(0,logprobs.shape[0],num_seq)]
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
            inputs.pop(dict_keys['input'])
            return {dict_keys['output']:gen,**inputs}
        else:
            return model.text_generation(inputs, **gen_kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(inps),max_workers)) as executor:
        if msg != '':
            out = list(tqdm(executor.map(tgi_generate,inps),total = len(inps),desc = msg))
        else:
            out = list(executor.map(tgi_generate,inps))
    return out


def get_extract_template(model_name,instruction_tuned): # get extract template (only used if TGI not used)
    if instruction_tuned:
        if 'neural-chat-7b' in model_name:
            extract_fn = lambda x: x.split("### Assistant:\n")[-1].strip()
        elif 'Mistral' in model_name or 'zephyr' in model_name:
            extract_fn = lambda x : x.split(" [/INST]")[-1].strip()
        elif 'Llama' in model_name:
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
        system_prompt = "You are a student who is eager to learn about new things." # only use if question generation
    else:
        system_prompt = ""
    
    if 'neural-chat-7b' in model_name or 'Llama' in model_name: # Mistral doesnt have system prompt
        sys_msg = [{'role':'system','content':system_prompt}]
        response = sys_msg + response
    if mode != 'label':
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
    if 'mistral' in model_name:
        return "<s>[INST] {instruction} [/INST] {output}</s>", "[/INST]"
    elif 'Llama' in model_name:
        return "<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>", "<|assistant|>"

def clean_question(question,len_fs):
    """
    given question in "question\n{digit}: question\n{digit+1} ..."
    return question, digit should be > len_fs
    """
    # if 'question:' in question.lower():
    #     take_pos = question.lower().find('question:') + len('question:')
    #     question = question[take_pos:].strip()
    match = re.search(r'(\d+):', question)
    if match:
        digit = int(match.group(1))  # Convert the found digits to an integer
        if digit > len_fs:
            # If the digit is above the threshold, extract everything before the digit
            return question[:match.start()].strip()
        else:
            return question.split('\n\n')[0].strip()
    else:
        return question.split('\n\n')[0].strip()

def clean_answer(answer):
    if 'Q:' in answer: # Clean off excess text
        answer = answer.split('Q:')[0].strip()
    else:
        answer.split('\n\n')[0].strip()
    return answer

def extract_str_in_bracket(x):
    pattern = r"\((.*?)\)"
    match = re.search(pattern, x)
    if match:
        extracted_text = match.group(1) 
        return extracted_text
    else:
        return x

def get_nouns_and_embeddings(embedder: SentenceTransformer) -> Dict:
    # nltk.download('omw-1.4')
    from nltk.corpus import wordnet as wn
    dir_storage = "data"
    os.makedirs(dir_storage, exist_ok=True)
    wordnet_data_path = dir_storage + "/wordnet_data.pkl"
    if os.path.exists(wordnet_data_path):
        with open(wordnet_data_path, "rb") as dump_handle:
            wordnet_data = pickle.load(dump_handle)
    else:
        all_nouns = []
        for synset in wn.all_synsets("n"):
            lemma_names = [str(lemma.name()) for lemma in synset.lemmas()]
            lemma_descs = [str(synset.definition()) for lemma in synset.lemmas()]
            lemms = [n + " ### " + d for n, d in zip(lemma_names, lemma_descs)]
            all_nouns.extend(lemms)
        all_nouns = list(set(all_nouns))
        all_embeddings = embedder.encode(all_nouns, device="cpu", convert_to_numpy=True)
        wordnet_data = {"all_nouns": all_nouns, "all_embeddings": all_embeddings}
        with open(wordnet_data_path,'wb') as dump_handle:
            pickle.dump(wordnet_data, dump_handle)
    return wordnet_data

def get_topic_embedding_space(all_embeddings: List) -> KDTree:
    all_embeddings = torch.tensor(all_embeddings)
    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=1, dim=-1)
    return KDTree(all_embeddings)

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