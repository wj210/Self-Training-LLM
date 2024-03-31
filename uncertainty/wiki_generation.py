import os
import json
import numpy as np
from huggingface_hub import InferenceClient
from tqdm import tqdm
from scorer import NLIScorer,async_process
import argparse
import torch
import pickle
import random
from utils import *
from data_utils import *
from templates import format_question_generation,format_question_w_document,format_answer_w_document
from functools import partial
from copy import deepcopy
from dataclasses import asdict
import yaml
from rouge_score import rouge_scorer
from types import SimpleNamespace

def open_generate_qns(client,scorer,topics,document_dict,max_workers,gen_kwargs,tokenizer,model_name,question_path=None,test_path=None,test_size=1000,qn_per_topic=1,use_tgi=True):
    is_instruct_tuned = if_instruction_tuned(model_name)
    max_document_length = 2000
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score_key = SCORE_KEY[scorer.scoring_method]
    dict_keys = {}
    
    def tgi_generate(prompt,prompt_type = 'topic'):
        return client.text_generation(prompt, **gen_kwargs[prompt_type])
    
    def generate_qns_prompt(topic,docu_dict):
        document = docu_dict[topic]
        if len(tokenizer.encode(document,add_special_tokens=False)) > max_document_length:
            document = tokenizer.decode(tokenizer.encode(document,add_special_tokens=False)[:max_document_length])
        qn_prompt = format_question_w_document(topic,document)
        if is_instruct_tuned:
            qn_prompt = [{'role':'user','content':qn_prompt}]
            qn_prompt = format_response(qn_prompt,model_name,tokenizer,mode='question')

        return {'topic':topic,'instruction':qn_prompt,'document':document}
    
    def generate_ans(qns_dicts):
        if use_tgi:
            qns_dicts = [qns_dicts]
        else:
            inp_batch = []
        for qns_dict in qns_dicts:
            qns = qns_dict['instruction']
            topic = qns_dict['topic']
            document = qns_dict['document']
            sample_ans_prompt = [{'role':'user','content':qns}]
            ref_ans_prompt = [{'role':'user','content':format_answer_w_document(qns,document)}]
            if is_instruct_tuned:
                sample_ans_prompt = format_response(sample_ans_prompt,model_name,tokenizer,mode='answer')
                ref_ans_prompt = format_response(ref_ans_prompt,model_name,tokenizer,mode='answer')
            else:
                sample_ans_prompt = sample_ans_prompt[0]['content']
                ref_ans_prompt = ref_ans_prompt[0]['content']
            if use_tgi:
                ref_answer = tgi_generate(ref_ans_prompt,prompt_type='ref_answer')
                sample_answer = tgi_generate(sample_ans_prompt,prompt_type='sample_answer')
                return {'sample_answer':sample_answer,'ref_answer':ref_answer,'instruction':qns,'topic':topic}
            else:
                inp_batch.append({'sample_answer':sample_ans_prompt,'ref_answer':ref_ans_prompt,'instruction':qns,'topic':topic})
        if not use_tgi:
            out_batch = [{'topic':x['topic'],'instruction':x['instruction']} for x in inp_batch]
            for ans_type in ['sample_answer','ref_answer']:
                i_batch = [{ans_type: x[ans_type]} for x in inp_batch]
                dict_keys['input'] = ans_type
                dict_keys['output'] = ans_type
                gen_batch = HF_generate(i_batch,client,tokenizer,gen_kwargs[ans_type],max_workers=len(i_batch),dict_keys=dict_keys,return_probs=True)
                for o,g in zip(out_batch,gen_batch):
                    o[ans_type] = g[ans_type]
            return out_batch 
        
    def get_scored_ds(ans_dict):
        all_qn_confidences = []
        for ans_d in tqdm(ans_dict,total = len(ans_dict),desc=f'Scoring questions based on {scorer.scoring_method}'):
            ref_ans = ans_d['ref_answer']
            sample_ans = ans_d['sample_answer']
            instruction = ans_d['instruction']
            score_dict = scorer.get_score(instruction,ref_ans,sample_ans)
            score_dict = {**ans_d,**score_dict}
            all_qn_confidences.append(score_dict)
        return all_qn_confidences
    
    if not os.path.exists(question_path):
        qns_prompts = [generate_qns_prompt(topic,document_dict) for topic in topics]
        if use_tgi:
            dict_keys['input'] = 'instruction'
            dict_keys['output'] = 'instruction'
        qns_dict = HF_generate(qns_prompts,client,tokenizer,gen_kwargs['questions'],use_tgi=use_tgi,max_workers=max_workers,dict_keys=dict_keys,msg = 'Generating questions')
        
        # Check for duplicates.# 
        if qn_per_topic > 1:
            non_duplicate_qns = []
            for q in qns_dict:
                n_instructions = q['instruction']
                if use_tgi:
                    n_instructions = [n_instructions.generated_text] + [qq.generated_text for qq in n_instructions.details.best_of_sequences]
                else:
                    n_instructions = sum(n_instructions,[])
                non_duplicate_qns.append({'topic':q['topic'],'document':q['document'],'instruction':n_instructions[0]}) # first include the 1st.
                
                # check with the rest of the instruction for duplicates
                checking_instr = [n_instructions[0]]
                for remaining_instr in n_instructions[1:]:
                    if not is_duplicate(r_scorer,remaining_instr,checking_instr,max_rouge=0.5):
                        non_duplicate_qns.append({'topic':q['topic'],'document':q['document'],'instruction':remaining_instr})
                        checking_instr.append(remaining_instr)
                        
            print (f'removed {(len(qns_dict)*qn_per_topic) - len(non_duplicate_qns)} duplicate questions')
        else:
            if use_tgi:
                q['instruction'] = q['instruction'].generated_text
            non_duplicate_qns = qns_dict
        
        ## Take qn from test_size topics to reserve as test set
        if not os.path.exists(test_path): # save heldout test qns
            test_set = []
            train_set = []
            topic_qn_count = defaultdict(int)
            for qn_d in non_duplicate_qns:
                topic_qn_count[qn_d['topic']] +=1
            unique_topics = [k for k,v in topic_qn_count.items() if v > 1] # we only consider the topics with more than 1 qn so we can take 1 for test set
            unique_topics = set(unique_topics[:test_size])

            for qn_d in non_duplicate_qns:
                if len(unique_topics) > 0:
                    if qn_d['topic'] in unique_topics:
                        test_set.append(qn_d)
                        unique_topics.remove(qn_d['topic'])
                    else:
                        train_set.append(qn_d)
                else:
                    train_set.append(qn_d)
                    
            with open(test_path,'w') as f:
                for instance in test_set:
                    json.dump(instance,f,ensure_ascii=False)
                    f.write('\n')
        else:
            train_set = non_duplicate_qns

        if use_tgi:
            ans_dict = async_process(generate_ans,train_set,max_workers,msg = 'Generating answers')
        else:
            ans_dict = []
            for batch_i in tqdm(range(0,len(train_set),max_workers),total = len(train_set)//max_workers,desc='Generating answers'):
                batch = train_set[batch_i:batch_i+max_workers]
                ans_batch = generate_ans(batch)
                ans_dict.extend(ans_batch)
        with open(question_path,'wb') as f:
            pickle.dump(ans_dict,f)
    else:
        with open(question_path,'rb') as f:
            ans_dict = pickle.load(f)

    if score_key not in ans_dict[0]: # not yet scored.
        all_qn_confidences = get_scored_ds(ans_dict)
        with open(question_path,'wb') as f: # re-update the question_set with other approach scores.
            pickle.dump(all_qn_confidences,f)
    else:
        all_qn_confidences = ans_dict
    # Split into unknown and known to generate dpo samples
    unknown_qns = return_question_type(all_qn_confidences,scorer.scoring_method)
    get_ds_fn = partial(scorer.get_dpo_sample,few_shots=None)
    generated_ds = []
    for unknwn_qn in tqdm(unknown_qns,desc='Generating dpo samples',total = len(unknown_qns)):
        try:
            generated_ds.append(get_ds_fn(unknwn_qn))
        except Exception as e:
            print (e)
    generated_ds = [t for t in generated_ds if t is not None]
    return generated_ds
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082,help = 'port for TGI')
    parser.add_argument("--config_path", type=str, default="",required=True)
    ## Dataset curation ##
    parser.add_argument("--answer_generator", type=str, default="self_fewshot",required=True)
    parser.add_argument("--scoring_method", type=str, default="BSDetector",required=True)
    parser.add_argument("--topic_generator", type=str, default="fixed_mmlu",required=True)
    parser.add_argument("--use_tgi", action='store_true',help = 'use TGI for loaded model to do eval')
    parser.add_argument("--max_response_tokens", type=int, default=256,help = 'max tokens for answer generation')
    parser.add_argument("--num_topics",  type = int,default = 200,help = 'total qns to generate')
    parser.add_argument("--num_concurrent_calls", type=int, default=32,help = 'max_api_calls to TGI at a time')
    parser.add_argument("--num_samples", type=int, default=5,help = 'number of sampled responses')
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--questions_per_topic",  type = int,default = 10)
    parser.add_argument("--test_size",  type = int,default = 500)
    parser.add_argument("--beta",  type = float,default = 1.0,help = 'Trade-off between observed and self-reflected confidence, for BSDetector only.')
    parser.add_argument("--answer_generator_port", type=int, default=8083,help = 'port for TGI to generate answer, only used for mistral_8x7 if loaded locally.')
    parser.add_argument("--openai_api_key_path",  type = str,default = '',help = 'a text file for openai api key, required only if using factscorer.')
    args = parser.parse_args()
    
    ## Seed ## 
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    ## Get config ##
    with open(args.config_path,'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
        
    if args.answer_generator in ['gpt4,gpt3.5']:
        assert args.openai_api_key_path != '','Need to provide openai api key for gpt4/gpt3.5'
        with open(args.openai_api_key_path,'r') as f:
            openai_api_key = f.read
        os.environ['OPENAI_API_KEY'] = openai_api_key
    
    ds_name = 'wiki'
    ds_config_path = f'configs/data/{ds_name}.yaml'
    with open(ds_config_path,'r') as f:
        ds_config = yaml.safe_load(f)
    
    if args.scoring_method == 'BSDetector':
        scoring_name = 'conf' 
    elif args.scoring_method == 'semantic_consistency':
        scoring_name = 'entropy'
    else:
        scoring_name = 'hallu'
    config.train_dataset_path = config.train_dataset_path.format(topic_generator=args.topic_generator,
                                                                 answer_generator=args.answer_generator,
                                                                scoring_name=scoring_name)
    ds_config['test_dataset_path'] = ds_config['test_dataset_path'].format(dataset_name = ds_name)
    if not os.path.exists(ds_config['test_dataset_path']):
        assert args.questions_per_topic >1, 'need more 1 qn per topic, to reserve test set.'
    
    ## Question path ##
    config.question_path = config.question_path.format(topic_generator=args.topic_generator)
    
    ## create dirs
    for required_paths in [config.train_dataset_path,ds_config['test_dataset_path'],config.question_path,]:
        base_dir_name = os.path.dirname(required_paths)
        os.makedirs(base_dir_name,exist_ok=True)
    
    config.topic_generator = args.topic_generator
    config.answer_generator = args.answer_generator
    config.scoring_method = args.scoring_method
    
    ## Generate kwargs
    gen_kwargs = {'topic':{'max_new_tokens':5, 'do_sample':True, 'temperature':1.0,'repetition_penalty':1.1},
                    'questions':{'max_new_tokens':ds_config.get('max_question_tokens',128), 'do_sample':True, 'temperature':1.0,'repetition_penalty':1.1,'details':True,'best_of':args.questions_per_topic},
                    'ref_answer':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':False, 'repetition_penalty':1.1,'details':True},
                    'sample_answer':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True, 'temperature':1.0,'best_of':args.num_samples,'repetition_penalty':1.1,'details':True}}
    
    # client
    if args.use_tgi:
        client = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    else:
        client = load_hf_model(config.model_name,quantized=True)
    
    base_tokenizer = load_tokenizer(config.model_name)

    # Get random topics with supporting document from wiki
    all_topics,sup_documents = get_wiki(args.num_topics,ds_config)

    # scorer
    scorer = NLIScorer(client,config.model_name,base_tokenizer,config.scoring_method,args.beta,max_response_tokens=ds_config.get('max_response_tokens',128),answer_generator=config.answer_generator,answer_generator_port=args.answer_generator_port,ref_as_chosen=True) # set ref_as_chosen as true.
    
    train_unknown_qns = open_generate_qns(client,
                                             scorer,
                                             all_topics,
                                             sup_documents,
                                             args.num_concurrent_calls,
                                             gen_kwargs,
                                             base_tokenizer,
                                             config.model_name,
                                             question_path=config.question_path,
                                             test_path = ds_config['test_dataset_path'],
                                             test_size = args.test_size,
                                             qn_per_topic = args.questions_per_topic,
                                             use_tgi=args.use_tgi)

    with open(config.train_dataset_path,'w') as f:
            for instance in train_unknown_qns:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    main()