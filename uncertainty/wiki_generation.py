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
from templates import format_question_w_document,format_answer,question_generation_examples
from topic_generator import get_embedding,get_related_topics,get_views_for_topic
from functools import partial
from copy import deepcopy
from dataclasses import asdict
import yaml
# from rouge_score import rouge_scorer
from types import SimpleNamespace
from multiprocessing import Pool
import spacy

def open_generate_qns(client,
                      scorer,
                      topics,
                      document_dict,
                      max_workers,
                      gen_kwargs,
                      tokenizer,
                      model_name,
                      question_path=None,
                      test_path=None,
                      test_topics = [],
                      qn_per_topic=1,
                      use_tgi=True,
                      question_filtering=False
                      ):
    
    is_instruct_tuned = if_instruction_tuned(model_name)
    ans_call_fn = async_process if use_tgi else batch_ops
    max_document_length = 1024
    chunk_length = 512
    sent_processor = spacy.load("en_core_web_sm")
    score_key = SCORE_KEY[scorer.scoring_method]
    dict_keys = {}
    max_few_shots = 10 if 'llama' not in model_name.lower() else 5 # llama max context length is 2048
    
    ## Craft few-shots for qn and answer.
    qn_few_shot = sum([[{'role':'user','content':format_question_w_document(fs['topic'],fs['document'])},
                    {'role':'assistant','content':fs['question']}] for fs in question_generation_examples],[])[:max_few_shots]
    ans_few_shot_ref = sum([[{'role':'user','content':format_answer(fs['question'],fs['document'])},
                    {'role':'assistant','content':fs['answer']}] for fs in question_generation_examples],[])[:max_few_shots]
    ans_few_shot_sample = sum([[{'role':'user','content':format_answer(fs['question'])}, # leave document out
                    {'role':'assistant','content':fs['answer']}] for fs in question_generation_examples],[])[:max_few_shots]
    
    def tgi_generate(prompt,prompt_type = 'topic'):
        return client.text_generation(prompt, **gen_kwargs[prompt_type])
    
    def generate_qns_prompt(topic,docu_dict,test_generation=False,num_few_shot=10):
        out_list = []
        document = docu_dict[topic]
        if test_generation:
            chunked_documents = [tokenizer.decode(tokenizer.encode(document,add_special_tokens=False)[:max_document_length])]
        else:
            chunked_documents = chunk_document(document,tokenizer,sent_processor,chunk_length,qn_per_topic)
        if len(chunked_documents) == 0:
            return None
        for doc in chunked_documents:
            qn_prompt = format_question_w_document(topic,doc)
            qn_prompt = [{'role':'user','content':qn_prompt}]
            qn_prompt = qn_few_shot[:int(num_few_shot*2)] + qn_prompt
            if is_instruct_tuned or test_generation:
                if not test_generation:
                    qn_prompt = format_response(qn_prompt,model_name,tokenizer,mode='question')
            else:
                qn_prompt = join_non_instruct(qn_prompt)
            out_list.append({'topic':topic,'instruction':qn_prompt,'document':doc})
        return out_list
    
    def generate_ans(qns_dicts,type_ ='ref_answer'):
        assert type_ in ['ref_answer','sample_answer','question_filtering'], 'Invalid type.'
        if use_tgi:
            qns_dicts = [qns_dicts]
        else:
            inp_batch = []
        for qns_dict in qns_dicts:
            qns = qns_dict['instruction']
            document = qns_dict['document']
            if type_ in ['ref_answer','question_filtering']:
                ans_fs = ans_few_shot_ref
            else:
                ans_fs = []
                document = ''
            ans_prompt = ans_fs + [{'role':'user','content':format_answer(qns,document)}]
            if is_instruct_tuned:
                ans_prompt = format_response(ans_prompt,model_name,tokenizer,mode='answer')
            else:
                ans_prompt = join_non_instruct(ans_prompt)
            if use_tgi:
                try:
                    ans = tgi_generate(ans_prompt,prompt_type=type_)
                    return {**qns_dict,type_:ans}
                except Exception as e:
                    print (e)
                    return None
            else:
                inp_batch.append({**qns_dict,type_:ans_prompt})
        if not use_tgi:
            ans_batch = [{type_:x[type_]} for x in inp_batch]
            dict_keys['input'] = type_
            dict_keys['output'] = type_
            generate_args = gen_kwargs[type_]
            gen_batch = HF_generate(ans_batch,client,tokenizer,generate_args,max_workers=len(ans_batch),dict_keys=dict_keys,return_probs=True)
            for o,g in zip(inp_batch,gen_batch):
                o[type_] = g[type_]
            return inp_batch 
        
    def get_scored_ds(ans_dict):
        all_qn_confidences = []
        for ans_d in tqdm(ans_dict,total = len(ans_dict),desc=f'Scoring questions based on {scorer.scoring_method}'):
            try:
                ref_ans = ans_d['ref_answer']
                if 'sample_answer' in ans_d:
                    sample_ans = ans_d['sample_answer']
                    return_full_dict = True
                else:
                    sample_ans = ans_d['question_filtering']
                    ans_d.pop('question_filtering')
                    return_full_dict = False
                instruction = ans_d['instruction']
                score_dict = scorer.get_score(instruction,ref_ans,sample_ans)
                if score_dict == None:
                    continue
                if return_full_dict:
                    score_dict = {**ans_d,**score_dict}
                else:
                    score_dict = {**ans_d,score_key:score_dict[score_key]} # just need the score.
                all_qn_confidences.append(score_dict)
            except Exception as e:
                print (e)
        return all_qn_confidences
    
    ###################
    ## GENERATE TEST ##
    ####################
    if not os.path.exists(test_path):
        assert len(test_topics) > 0,'No test topics to generate questions for.'
        test_qn_prompts = [generate_qns_prompt(t,document_dict,test_generation=True) for t in test_topics] # just set 1 few-shot
        test_qn_prompts = [q for q in test_qn_prompts if q is not None]
        test_qn_prompts = sum(test_qn_prompts,[]) # flatten
        test_ds = []
        total_cost = 0.
        for tp in tqdm(test_qn_prompts,total = len(test_qn_prompts),desc = 'Generating test questions'):
            num_tries = 0
            while num_tries < 5:
                test_qn,cost = openai_call('gpt-3.5-turbo-instruct',tp['instruction'],max_tokens=gen_kwargs['questions']['max_new_tokens'],temperature=gen_kwargs['questions']['temperature'])
                total_cost += cost
                if check_question(test_qn):
                    break
                else:
                    num_tries += 1
            if test_qn is not None:
                test_ds.append({'topic':tp['topic'],'document':tp['document'],'instruction':test_qn})
        print ('Total cost:',total_cost)
        with open(test_path,'w') as f:
            for instance in test_ds:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')
    
    if not os.path.exists(question_path):
        #######################
        ## GENERATE QUESTION ##
        #######################
        qns_prompts_dict = [generate_qns_prompt(topic,document_dict) for topic in topics]
        qns_prompts = [q for q in qns_prompts_dict if q is not None]
        qns_prompts = sum(qns_prompts,[]) # flatten
        if use_tgi:
            dict_keys['input'] = 'instruction'
            dict_keys['output'] = 'instruction'
        qns_dict = HF_generate(qns_prompts,client,tokenizer,gen_kwargs['questions'],use_tgi=use_tgi,max_workers=max_workers,dict_keys=dict_keys,msg = 'Generating questions')
        checked_qns_dict = []
        for q in qns_dict: # clean the question.
            if check_question(q['instruction']):
                q['instruction'] = clean_question(q['instruction'])
                checked_qns_dict.append(q)
        print (f'Removed {len(qns_dict)-len(checked_qns_dict)} questions due to bad quality.')
        qns_dict = checked_qns_dict
        
        #######################
        ## GENERATE ANSWER ##
        #######################
        ref_ans_fn = partial(generate_ans,type_='ref_answer')
        ref_ans_content = filter_none(ans_call_fn(ref_ans_fn,qns_dict,max_workers,msg = 'Generating ref answers'))
        ref_ans_content = check_answer(ref_ans_content,key='ref_answer',use_tgi=use_tgi) # filter the answer for any "i cannot answer"
        if question_filtering:
            qn_filter_fn = partial(generate_ans,type_='question_filtering')
            qn_filter_ans = filter_none(ans_call_fn(qn_filter_fn,ref_ans_content,max_workers,msg = 'Generating hallucination score to filter questions'))
            scored_qns = get_scored_ds(qn_filter_ans) # get question quality score
            filtered_qns = return_question_type(scored_qns,scorer.scoring_method,'known') # filter out poor questions
            print (f'Filtered down to {len(filtered_qns)} from {len(scored_qns)} questions.')
            for q in filtered_qns:
                q.pop(score_key) # Impt, to remove the score from the question else later will skip the scored_ds fn.
        else:
            filtered_qns = ref_ans_content
        sample_ans_fn = partial(generate_ans,type_='sample_answer')
        ans_dict = filter_none(ans_call_fn(sample_ans_fn,filtered_qns,max_workers,msg = 'Generating sample answers'))
        with open(question_path,'wb') as f:
            pickle.dump(ans_dict,f)
    else:
        with open(question_path,'rb') as f:
            ans_dict = pickle.load(f)
    
    ####################
    ## GENERATE SCORE ##
    ####################    
    if score_key not in ans_dict[0]: 
        all_qn_confidences = get_scored_ds(ans_dict)
        with open(question_path,'wb') as f: 
            pickle.dump(all_qn_confidences,f)
    else:
        all_qn_confidences = ans_dict
        
    #########################
    ## GENERATE DPO SAMPLE ##
    #########################
    unknown_qns = return_question_type(all_qn_confidences,scorer.scoring_method,'unknown')
    print (f'Remaining {len(unknown_qns)} unknown questions out of {len(all_qn_confidences)} questions.')
    get_ds_fn = partial(scorer.get_dpo_sample,fs_messages=ans_few_shot_sample)
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
    parser.add_argument("--topic_generator", type=str, default="wiki",required=True)
    parser.add_argument("--use_tgi", action='store_true',help = 'use TGI for loaded model to do eval')
    parser.add_argument("--max_response_tokens", type=int, default=256,help = 'max tokens for answer generation')
    parser.add_argument("--num_topics",  type = int,default = 200,help = 'total qns to generate')
    parser.add_argument("--num_samples", type=int, default=5,help = 'number of sampled responses')
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--questions_per_topic",  type = int,default = 10)
    parser.add_argument("--test_size",  type = int,default = 300)
    parser.add_argument("--beta",  type = float,default = 1.0,help = 'Trade-off between observed and self-reflected confidence, for BSDetector only.')
    parser.add_argument("--answer_generator_port", type=int, default=8083,help = 'port for TGI to generate answer, only used for mistral_8x7 if loaded locally.')
    parser.add_argument("--question_filtering",  action = 'store_true',help= 'to filter question based on hallucination score.')
    parser.add_argument("--openai_api_key_path",  type = str,default = 'openai_api_key.txt',help = 'a text file for openai api key, required only if using factscorer.')
    args = parser.parse_args()
    
    args.num_concurrent_calls = np.floor(320/args.num_samples).astype(int).item() # set max request at 320.
    
    ## Seed ## 
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    ## Get config ##
    with open(args.config_path,'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
    
    ds_config_path = f'configs/data/wiki.yaml'
    with open(ds_config_path,'r') as f:
        ds_config = yaml.safe_load(f)
    
    if args.scoring_method == 'BSDetector':
        scoring_name = 'conf' 
    elif args.scoring_method == 'semantic_consistency':
        scoring_name = 'entropy'
    else:
        scoring_name = 'hallu'
    ## Training path ##
    config.train_dataset_path = config.train_dataset_path.format(topic_generator=args.topic_generator,
                                                                 answer_generator=args.answer_generator,
                                                                scoring_name=scoring_name)
    if args.question_filtering:
        config.train_dataset_path = config.train_dataset_path.replace('.jsonl','_qf.jsonl')
    ## Test path ##
    ds_config['test_dataset_path'] = ds_config['test_dataset_path'].format(dataset_name = args.topic_generator)
    if args.answer_generator in ['gpt4,gpt3.5'] or not os.path.exists(ds_config['test_dataset_path']):
        assert args.openai_api_key_path != '','Need to provide openai api key for gpt4/gpt3.5'
        with open(args.openai_api_key_path,'r') as f:
            openai_api_key = f.read()
        if openai_api_key == '':
            raise ValueError('Need to provide openai api key for gpt4/gpt3.5')
        os.environ['OPENAI_API_KEY'] = openai_api_key
    
    ## Question path ##
    config.question_path = config.question_path.format(topic_generator=args.topic_generator)
    if args.question_filtering:
        config.question_path = config.question_path.replace('.pkl','_qf.pkl')
    
    ## create dirs
    for required_paths in [config.train_dataset_path,ds_config['test_dataset_path'],config.question_path,]:
        base_dir_name = os.path.dirname(required_paths)
        os.makedirs(base_dir_name,exist_ok=True)
    
    config.topic_generator = args.topic_generator
    config.answer_generator = args.answer_generator
    config.scoring_method = args.scoring_method
    
    ## Generate kwargs
    gen_kwargs = {'topic':{'max_new_tokens':5, 'do_sample':True, 'temperature':1.0,'repetition_penalty':1.1},
                    'questions':{'max_new_tokens':ds_config.get('max_question_tokens',128), 'do_sample':False,'repetition_penalty':1.1},
                    'question_filtering':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True, 'temperature':0.5,'best_of':args.num_samples,'repetition_penalty':1.1,'details':True},
                    'ref_answer':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':False, 'repetition_penalty':1.1,'details':True},
                    'sample_answer':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True, 'temperature':1.0,'best_of':args.num_samples,'repetition_penalty':1.1,'details':True}}
    
    # client
    if args.use_tgi:
        client = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    else:
        client = load_hf_model(config.model_name,quantized=True)
    
    base_tokenizer = load_tokenizer(config.model_name)
    if not args.use_tgi:
        base_tokenizer.padding_side = 'left'
    # Get random topics with supporting document from wiki
    embedding_topic_path = 'data/embeddings/wiki.pkl'
    accepted_topic_doc_path = 'data/embeddings/accepted_topic_doc.pkl'
    os.makedirs('data/embeddings',exist_ok=True)
    if not os.path.exists(accepted_topic_doc_path): ## get documents longer than a certain length.
        topic2docu = {}
        wiki_ds = get_wiki(-1,ds_config,get_ds=True)
        for d in tqdm(wiki_ds,total=len(wiki_ds)):
            if len(d['text'].split()) > 512:
                topic2docu[d['title']] = d['text']
        with open(accepted_topic_doc_path,'wb') as f:
            pickle.dump(topic2docu,f)
    else:
        with open(accepted_topic_doc_path,'rb') as f:
            topic2docu = pickle.load(f)
        
    if not os.path.exists(embedding_topic_path):
        wiki_topics = list(topic2docu.keys())
        embedding_dict = get_embedding(wiki_topics,embedding_topic_path)
    else:
        with open(embedding_topic_path,'rb') as f:
            embedding_dict = pickle.load(f)

    if config.topic_generator == 'wiki':
        if os.path.exists(ds_config['test_dataset_path']): # already exist, no need gather test topics.
            embedding_dict = None
        all_topics,sup_documents,test_topics = get_wiki(args.num_topics,ds_config,num_test_topics= args.test_size,embedding_dict=embedding_dict,topic2docu=topic2docu)
    else: # get topics based on predefined topics like in MMLU.
        assert config.topic_generator == 'wiki_mmlu','Only mmlu is supported for now.' #TODO support more benchmarks providing useful topics.
        query_topics = get_fixed_topics(config.topic_generator)
        for i,t in enumerate(query_topics):
            if '_' in t:
                query_topics[i] = t.replace('_',' ')
        if not os.path.exists(ds_config['test_dataset_path']):
            to_take = args.num_topics + args.test_size
        else:
            to_take = args.num_topics
        
        k_per_topic = np.ceil(to_take/(len(query_topics))).astype(int).item()
        all_topics = get_related_topics(embedding_dict,query_topics,k_per_topic)
        sup_documents = {k:topic2docu[k] for k in all_topics if k in topic2docu}
        accepted_topics = list(sup_documents.keys())
        if len(accepted_topics) != len(all_topics):
            print (f'Gathered {len(accepted_topics)} out of {len(all_topics)} required.')
        all_topics = accepted_topics
        if not os.path.exists(ds_config['test_dataset_path']):
            random.shuffle(all_topics)
            test_topics = all_topics[:args.test_size]
            all_topics = all_topics[args.test_size:]
        else:
            test_topics = []

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
                                             test_topics = test_topics,
                                             qn_per_topic = args.questions_per_topic,
                                             use_tgi=args.use_tgi,
                                             question_filtering = args.question_filtering
                                             )

    with open(config.train_dataset_path,'w') as f:
        for instance in train_unknown_qns:
            json.dump(instance,f,ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    main()