from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
import re
from peft import LoraConfig, PeftModel,AutoPeftModelForCausalLM
from collections import defaultdict
import concurrent.futures
from functools import partial
import os
from sentence_transformers import SentenceTransformer
import nltk
import pickle
from nltk.corpus import wordnet as wn
from typing import List, Dict
from scipy.spatial import KDTree

def load_hf_model(model_name,model_path=None,use_tgi=False): ## if use tgi, dont quantize.
    if not use_tgi:
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

    if model_path is not None:
        file_names = os.listdir(model_path)
        if 'adapter_model.safetensors' in file_names: # if not merged, we merge it.
            peft_model = PeftModel.from_pretrained(
                model=base_model,
                model_id = model_path,
                is_trainable=False,
            )
            base_model = peft_model.merge_and_unload()
    return base_model

def load_tokenizer(model_name,padding_side = "left"):
    tokenizer =  AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side = padding_side)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
    

def tgi_to_gen_kwargs(gen_kwargs): # convert TGI kwargs to HF kwargs
    if 'details' in gen_kwargs:
        gen_kwargs.pop('details')
    if 'best_of' in gen_kwargs:
        gen_kwargs['num_return_sequences'] = gen_kwargs.pop('best_of')
    gen_kwargs['return_dict_in_generate'] = True
    gen_kwargs['output_scores'] = True
    return gen_kwargs

def HF_generate(inps,model,tokenizer,gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=False): # basic HF stype generate, with option of using TGI
    if not use_tgi:
        tokenized_inps = tokenizer(inps, padding='longest', return_tensors="pt",truncation=False).to(model.device)
        gen_kwargs = {'return_dict_in_generate':True,'output_scores':True, **gen_kwargs}
        with torch.no_grad():
            model_outputs = model.generate(**tokenized_inps, **gen_kwargs)
        transition_scores = model.compute_transition_scores(model_outputs.sequences, model_outputs.scores, normalize_logits=True) # Log probs of transitions
        decoded =  tokenizer.batch_decode(model_outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [{'text':extract_ans_fn(d),'logprobs':lp} for d,lp in zip(decoded,transition_scores.detach().cpu().numpy())]
    else:
        return tgi_generate(inps,model,gen_kwargs)

def tgi_generate(inps,model,gen_kwargs): # Using TGI to generate
    def tgi_generate(prompt):
        return model.text_generation(prompt, **gen_kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(inps)) as executor:
        out = list(executor.map(tgi_generate,inps))
    return out


def get_prompt_and_extract_template(model_name): # get input template and decode extraction (not needed for TGI)
    """
    Prompt_type = [answer_gen, question_gen]
    answer_gen = no system prompt. User prompt is the only input
    question_gen include a system prompt to remind the model that it is a student and is curious.
    """
    system_prompt = "You are a student who is eager to learn about new things."
    template_dict = defaultdict(dict)
    if 'neural-chat-7b' in model_name:
        template_dict['prompt_fn']['answer_gen'] = lambda x: f"### System:\n### User:\n{x}\n### Assistant:\n"
        template_dict['prompt_fn']['question_gen'] = lambda x: f"### System:\n{system_prompt}\n### User:\n{x}\n### Assistant:\n"
        template_dict['extract_ans_fn'] = lambda x: x.split("### Assistant:\n")[-1].strip()
    elif 'Mistral' in model_name:
        template_dict['prompt_fn']['answer_gen'] = lambda x: f"<s>[INST] {x} [/INST]"
        template_dict['prompt_fn']['question_gen'] = lambda x: "<s>[INST] You are a student who is eager to learn about new things. [/INST]"+ "I am a student who is eager to learn about new things. I am aware of my lack of knowledge about some things.</s> " + f"[INST] {x} [/INST]"
        template_dict['extract_ans_fn'] = lambda x : x.split(" [/INST]")[-1].strip()
    elif 'Llama' in model_name:
        template_dict['prompt_fn']['answer_gen'] =  lambda x: f"<|system|>\n<|user|>\n{x}\n<|assistant|>\n"
        template_dict['prompt_fn']['question_gen'] =  lambda x: f"<|system|>\nYou are a student who is eager to learn about new things.</s>\n<|user|>\n{x}</s>\n<|assistant|>\n"
        template_dict['extract_ans_fn'] = lambda x: x.split("<|assistant|>\n")[-1].strip()
    return template_dict

def extract_str_in_bracket(x):
    pattern = r"\((.*?)\)"
    match = re.search(pattern, x)
    if match:
        extracted_text = match.group(1) 
        return extracted_text
    else:
        return x

def get_nouns_and_embeddings(embedder: SentenceTransformer) -> Dict:
    nltk.download('omw-1.4')
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