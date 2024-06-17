import os
import json
import numpy as np
from huggingface_hub import InferenceClient
from tqdm import tqdm
from scorer import NLIScorer,async_process
import argparse
import pickle
from utils import *
from data_utils import *
from templates import format_question_w_document,format_answer,question_generation_examples,format_message,prompt_format,generation_examples
from functools import partial
import yaml
# from rouge_score import rouge_scorer
from types import SimpleNamespace
from multiprocessing import Pool
import spacy
from copy import deepcopy

def open_generate_qns(client,
                      scorer, 
                      topics, # list of topics to generate qn
                      document_dict, # {topic: document} dict
                      max_workers, # batch size
                      gen_kwargs, 
                      tokenizer,
                      model_name,
                      question_path=None, # qn path to save/load
                      answer_path = None, # answer path to store sampled/gold answers
                      context_answer_path = None, # for instructions with context
                      test_path=None, # for held-out test set
                      test_topics = [], # topics for test set
                      qn_per_topic=1, 
                      use_tgi=True, 
                      generate_sft = False,
                      ref_as_chosen=True,
                      ):
    
    async_call_fn = async_process if use_tgi else batch_ops
    sent_processor = spacy.load("en_core_web_sm")
    question_model = 'gpt-3.5-turbo-0125'
    score_key = SCORE_KEY[scorer.scoring_method]
    chunk_length = 512
    num_test_topic= 20 # total 20*10 = 200
    test_qn_per_topic = 2
    openai_fs = 3
    ans_fs = 10
    
    if generate_sft:
        base = True
        add_few_shot = True
        add_document = False
    else:
        assert os.path.exists(question_path),'Reuse questions from sft phrase'
        base = False
        add_few_shot = False
        add_document = True
    
    dict_keys = {} # only use if not using TGI
    
    def tgi_generate(prompt,gen_type = 'greedy_answer'):
        return client.text_generation(prompt, **gen_kwargs[gen_type])
    
    # if not os.path.exists(test_path): ## randomly select topics from training set and make test data from the 1st test_chunk_length tokens of the document.
    #     category_2_topic = defaultdict(list)
    #     for t in topics:
    #         category_2_topic[t[0]].append(t[1])
    #     test_topics = [] # TODO once in-distribution works, try out-distribution for generalization.
    #     for cat, cat_topics in category_2_topic.items():
    #         long_topics = [t for t in cat_topics if len(tokenizer.encode(document_dict[t])) > test_chunk_length + (chunk_length*(qn_per_topic+1))]
    #         max_topics = min(num_test_topic,len(long_topics))
    #         if max_topics == 0:
    #             continue
    #         random_cat_topics = random.sample(long_topics,max_topics)
    #         test_topics.extend([(cat,topic) for topic in random_cat_topics])
    #     leftout_test_topics = set([t[1] for t in test_topics])
    # else:
    #     with open(test_path,'r') as f:
    #         test_ds = [json.loads(l) for l in f]
    #     leftout_test_topics = set([d['topic'] for d in test_ds])
    
    def generate_qns_prompt(topic_tuple,docu_dict,test_generation=False,existing_test_questions=None,existing_questions = {}):
        out_list = []
        total_cost = 0.
        topic = topic_tuple[1]
        category = topic_tuple[0]
        document = docu_dict[topic]
        if test_generation:
            chunked_documents = chunk_document(document,tokenizer,sent_processor,chunk_length,test_qn_per_topic)
        else:
            # if topic in leftout_test_topics:
            #     document = chunk_document(document,tokenizer,sent_processor,test_chunk_length,1,return_leftover =True) # return part that do not overlap with test
            #     if document is None:
            #         return None,None
            chunked_documents = chunk_document(document,tokenizer,sent_processor,chunk_length,qn_per_topic)
            ## Check if question already exist
            # assert existing_test_questions is not None, 'Need to generate test qns first.'
            # if topic in set([tq['topic'] for tq in existing_test_questions]):
            #     unique_qns = set([tq['instruction'] for tq in existing_test_questions if tq['topic'] == topic])
            # else:
            unique_qns = set()
            
        if chunked_documents == None:
            return None,None
    
        for doc in chunked_documents:
            question_fs = deepcopy(generation_examples)
            # if not test_generation and len(existing_questions[category]) > 0: # increase diversity.
            #     question_fs.extend(existing_questions[category])
            random.shuffle(question_fs)
            qn_few_shot = [{'instruction':format_question_w_document(fs['topic'],fs['document']),'answer':fs['instruction']} for fs in question_fs[:openai_fs]]
            qn_few_shot.append({'instruction':format_question_w_document(topic,doc)})
            qn_prompt = format_message(qn_few_shot,None,base=False,add_prompt = 'question',return_as_list=True)
            if not test_generation:
                qn_tries = 0
                while qn_tries < 3:
                    generated_qn,cost = openai_call(question_model,qn_prompt,gen_kwargs['questions']['max_new_tokens'],temperature=gen_kwargs['questions']['temperature'],n=1)
                    if generated_qn is None:
                        qn_tries += 1
                    else:
                        total_cost += cost
                        # generated_qn = refine_question(generated_qn)
                        if check_question(generated_qn,topic) and generated_qn not in unique_qns:
                            out_list.append({'topic':topic,'category':category,'instruction':generated_qn,'document':doc})
                            unique_qns.add(generated_qn)
                            break
                        qn_tries += 1
            else:
                out_list.append({'topic':topic,'category':category,'instruction':qn_prompt,'document':doc})
            
        return out_list,total_cost

    def generate_ans(qns_dicts,type_ ='gold_answer',use_chatgpt=False):
        assert type_ in ['gold_answer','gold_answer_sample','raw_answer','raw_answer_sample'], 'Invalid type.'
        if use_tgi:
            qns_dicts = [qns_dicts]
        else:
            inp_batch = []
        for qns_dict in qns_dicts:
            qns = qns_dict['instruction']
            document = qns_dict['document'].strip()
            topic = qns_dict['topic']
            category = qns_dict['category']
            if add_few_shot or use_chatgpt:
                fs_ans = generation_examples
                random.shuffle(fs_ans)
                if use_chatgpt:
                    fs_ans = fs_ans[:openai_fs]
                else:
                    fs_ans = fs_ans[:ans_fs] 
                if ('gold' not in type_ or not add_document) and not use_chatgpt:
                    fs_examples = [{'instruction':format_answer(fs['instruction'],fs['topic'],base=base),'answer':fs['answer']} for fs in fs_ans]
                else:
                    fs_examples = [{'instruction':format_answer(fs['instruction'],fs['topic'],fs['document'],base=base if not use_chatgpt else False,is_chat = use_chatgpt),'answer':fs['answer']} for fs in fs_ans]
            else:
                fs_examples = []
            if ('gold' not in type_ or not add_document) and not use_chatgpt:
                document = ''
            
            ans_prompt = fs_examples + [{'instruction':format_answer(qns,topic,document,base=base if not use_chatgpt else False)}]
            
            if not use_chatgpt:
                ans_prompt = format_message(ans_prompt,tokenizer,base=base)
            else:
                ans_prompt = format_message(ans_prompt,None,base=False,add_prompt = 'answer',return_as_list=True)
            total_cost = 0.0

            if use_tgi:
                try:
                    if type_ == 'gold_answer':
                        ans_generation_tries = 0
                        while ans_generation_tries < 5:
                            if use_chatgpt:
                                ans,cost = openai_call(question_model,ans_prompt,gen_kwargs['gold_answer']['max_new_tokens'],temperature=0.3,n=1)
                                total_cost += cost
                                if 'document' not in ans:
                                    return {**qns_dict,type_:ans},total_cost
                            else:
                                ans = tgi_generate(ans_prompt,type_)
                                ans = check_single_answer(ans,sent_processor,tokenizer,base=base,max_length =gen_kwargs['gold_answer']['max_new_tokens'])
                                if ans is not None:
                                    return {**qns_dict,type_:ans}
                                elif ans is None and not generate_sft: # for fine-tuned models, no need to do sampling.
                                    return None
                            ans_generation_tries += 1
                            if ans_generation_tries == 5:
                                return None
                    else:
                        ans = tgi_generate(ans_prompt,type_)
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
    
    def get_openai_answer(qns_dict,engine = 'gpt-3.5-turbo-0125',type_ = 'gold_answer'):
        assert type_ in ['gold_answer','gold_answer_sample'], 'Invalid type.'
        qns = qns_dict['instruction']
        ans_prompt = [{'role':'user','content':qns}]
        if type_ == 'gold_answer':
            temp = 0
            n = 1
            max_tokens = gen_kwargs['gold_answer']['max_new_tokens']
        else:
            temp = gen_kwargs[type_]['temperature']
            n = gen_kwargs[type_].get('best_of',1)
            max_tokens = gen_kwargs[type_]['max_new_tokens']
        openai_ans,cost = openai_call(engine,ans_prompt,max_tokens,temperature=temp,n=n)
        if openai_ans is not None:
            return {**qns_dict,type_:openai_ans},cost
        else:
            return None
        
    def get_scored_ds(ans_dict,compare_type = 'raw_answer_sample'):
        all_qn_confidences = []
        for ans_d in tqdm(ans_dict,total = len(ans_dict),desc=f'Scoring questions based on {scorer.scoring_method}'):
            try:
                ref_ans = ans_d['gold_answer']
                if compare_type == 'raw_answer_sample':
                    return_full_dict = True
                    compute_sample_scores = True
                else:
                    return_full_dict = False
                    compute_sample_scores = False
                sample_ans = ans_d[compare_type]
                instruction = ans_d['instruction']
                score_dict = scorer.get_score(instruction,ref_ans,sample_ans,compute_sample_scores=compute_sample_scores,base=base)
                if score_dict == None:
                    continue
                if return_full_dict:
                    scored_dict = {**ans_d,**score_dict}
                else:
                    scored_dict = {**ans_d,score_key:score_dict[score_key]}
                    if scorer.scoring_method == 'BSDetector': # include further scores to evaluate pref samples later.
                        scored_dict['gold_answer_scores'] = score_dict['all_nli_scores'][:,1] # take contradict scores.
                    elif scorer.scoring_method == 'SelfCheckGPT':
                        scored_dict['gold_answer_scores'] = score_dict['all_hallu_scores']
                all_qn_confidences.append(scored_dict)
            except Exception as e:
                print (e)
        return all_qn_confidences
    
    ###################
    ## GENERATE TEST ##
    ####################
    total_cost = 0.
    if os.path.exists(test_path):
        with open(test_path,'r') as f:
            test_ds = [json.loads(l) for l in f]
        existing_test_topics = set([d['topic'] for d in test_ds])
        
    else:
        test_ds = []
        existing_test_topics = set()
        
    scorer.use_tgi = False # set to false first as gpt4 doesn't produce a textgeneration item
    assert len(test_topics) > 0,'No test topics to generate questions for.'
    test_qn_prompts = [generate_qns_prompt(t,document_dict,test_generation=True) for t in test_topics]
    test_qn_prompts = [q for q in test_qn_prompts if q is not None]
    test_qn_prompts = [q[0] for q in test_qn_prompts]
    test_qn_prompts = sum(test_qn_prompts,[]) # flatten
    
    generated_test_questions = True if len(test_ds) == len(test_qn_prompts) else False
    generated_test_answers = True if ('greedy_response' in test_ds[0] and generated_test_questions) else False
    if not generated_test_questions:
        for tp in tqdm(test_qn_prompts,total = len(test_qn_prompts),desc = 'Generating test questions'):
            category = tp['category']
            topic = tp['topic']
            if topics in existing_test_topics:
                continue
            test_qn,cost = openai_call("gpt-4-turbo-2024-04-09",
                                        tp['instruction'],
                                        max_tokens=gen_kwargs['questions']['max_new_tokens'],
                                        temperature=0)
            total_cost += cost
            test_ds.append({'topic':topic,'category':category,'instruction':test_qn,'document':tp['document']})
        with open(test_path,'w') as f:
            for instance in test_ds:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')
        print (f'Total cost of generating questions for {len(test_ds)}: {total_cost:.3f}')
    if not generated_test_answers:
        new_test_ds = []
        for tp in tqdm(test_ds,total = len(test_ds),desc = 'Generating test answers'):
            qn_msg = [{'role':'user','content':tp['instruction']}]
            test_ans,cost = openai_call(question_model,
                                    qn_msg,
                                    max_tokens=gen_kwargs['gold_answer']['max_new_tokens'],
                                    temperature=0.)
            total_cost += cost
            if test_ans is not None:
                tp['greedy_response'] = test_ans
                new_test_ds.append(tp)
        test_ds = new_test_ds
        with open(test_path,'w') as f:
            for instance in test_ds:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')
        print (f'Total cost of generating answers for {len(test_ds)}: {total_cost:.3f}')
    scorer.use_tgi = use_tgi # **IMPT to set back

    all_costs = 0.
    #######################
    ## GENERATE QUESTION ##
    #######################
    # predicted_qn = defaultdict(list)
    random.shuffle(topics)
    
    if not os.path.exists(question_path):
        unique_qns = []
        question_gen_fn = partial(generate_qns_prompt,docu_dict=document_dict,existing_test_questions=test_ds)
        unique_qn_items = async_call_fn(question_gen_fn,topics,10,msg = 'Generating questions')
        unique_qns = sum([q[0] for q in unique_qn_items if q[0] is not None],[])
        all_costs += sum([q[1] for q in unique_qn_items if q[1] is not None])
        # for topic in tqdm(topics,total = len(topics),desc = 'Generating questions'):
        #     category = topic[0]
        #     curr_qns,cost = generate_qns_prompt(topic,document_dict,existing_test_questions=test_ds,existing_questions= predicted_qn)
        #     if curr_qns is not None:
        #         unique_qns.extend(curr_qns)
        #         predicted_qn[category].extend(curr_qns)
        #         all_costs += cost

        with open(question_path,'w') as f:
            for instance in unique_qns:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')
        print (f'Successfully generate {len(unique_qns)} out of {len(topics)*qn_per_topic} questions, Cost: {all_costs:.3f}')
        
    else:
        print ('Questions already existed, skipping question generating....')
        with open(question_path,'r') as f:
            unique_qns = [json.loads(l) for l in f]    
    
    ## generate a subset of answers with context for training sft ##
    if generate_sft and scorer.answer_generator == 'self' and ref_as_chosen: # if using gpt3.5 to generate label, do not need this.
        if not os.path.exists(context_answer_path):
            num_question_context = len(unique_qns)//3
            random.shuffle(unique_qns)
            context_questions = unique_qns[:num_question_context]
            context_ans_fn = partial(generate_ans,type_='gold_answer',use_chatgpt=True)
            context_ans = filter_none(async_call_fn(context_ans_fn,context_questions,max_workers,msg = 'Generating context answers'))
            total_cost = sum([c[1] for c in context_ans])
            context_ans = [c[0] for c in context_ans]
            print (f'Generated {len(context_ans)} answers with context out of {num_question_context}')
            all_costs += total_cost
            with open(context_answer_path,'w') as f:
                for instance in context_ans:
                    json.dump(instance,f,ensure_ascii=False)
                    f.write('\n')
            context_success_qn = set([c['instruction'] for c in context_ans])
            unique_qns = [q for q in unique_qns if q['instruction'] not in context_success_qn]
            print (f'Total cost for openai generation: {all_costs:.3f}')
        else:
            with open(context_answer_path,'r') as f:
                context_ans = [json.loads(l) for l in f]
            context_success_qn = set([c['instruction'] for c in context_ans])
            unique_qns = [q for q in unique_qns if q['instruction'] not in context_success_qn]
    
    if not os.path.exists(answer_path):
        #######################
        ## GENERATE ANSWER ##
        #######################
        if 'self' in scorer.answer_generator:
            gold_ans_fn = partial(generate_ans,type_='gold_answer')
            gold_ans_output = filter_none(async_call_fn(gold_ans_fn,unique_qns,max_workers,msg = 'Generating ref answers'))
            if not use_tgi:
                gold_ans_content = check_answer(gold_ans_output,sent_processor,tokenizer,key='gold_answer',use_tgi=use_tgi,base=base)
                print (f'{len(gold_ans_content)} gold answers out of {len(gold_ans_output)}')
            else:
                gold_ans_content = gold_ans_output
                print (f'{len(gold_ans_content)} gold answers out of {len(unique_qns)}')
        elif 'gpt' in scorer.answer_generator:
            gpt_model = 'gpt3.5' if '3.5' in scorer.answer_generator else 'gpt4'
            gpt_answer_path = f'data/wiki/{gpt_model}_answers.jsonl'
            if not os.path.exists(gpt_answer_path):
                gold_ans_fn = partial(get_openai_answer,engine=question_model,type_='gold_answer')
                gold_ans_content = filter_none(async_call_fn(gold_ans_fn,unique_qns,5,msg = f'Generating {scorer.answer_generator} ref answers'))
                openai_answer_cost = sum([g[1] for g in gold_ans_content])
                print (f'Total cost of generating {len(unique_qns)} answers using {scorer.answer_generator}: ',openai_answer_cost)            
                gold_ans_content = [g[0] for g in gold_ans_content]
                with open(gpt_answer_path,'w') as f:
                    for instance in gold_ans_content:
                        json.dump(instance,f,ensure_ascii=False)
                        f.write('\n')
            else:
                with open(gpt_answer_path,'r') as f:
                    gold_ans_content = [json.loads(l) for l in f]
            
        else:
            raise ValueError('Invalid answer generator.')
        
        if generate_sft:
            if scorer.answer_generator == 'self':
                with open(answer_path,'wb') as f: ## SFT only need few-shot labels
                    pickle.dump(gold_ans_content,f)
            exit(f'Successfully generate {len(gold_ans_content)} samples for SFT')

        if 'self' in scorer.answer_generator and ref_as_chosen and gen_kwargs['raw_answer_sample'].get('best_of',1) > 1:
            qn_filter_fn = partial(generate_ans,type_='gold_answer_sample') # generate to estimate confidence bounds around gold answer
            gold_sample_ans = filter_none(async_call_fn(qn_filter_fn,gold_ans_content,max_workers,msg = 'Generating answers to filter questions'))
            scored_qns = get_scored_ds(gold_sample_ans,compare_type = 'gold_answer_sample') # get question quality score
            question_key = 'question_hallucination'
            for q in scored_qns:
                q[question_key] = q.pop(score_key) # change to new key to not confuse with answer hallucination later.
        else:
            scored_qns = gold_ans_content
            
            ## Check if existing sample answers with different answer_generator exist, if yes, re-use the sampled answers since questions same, only chosen different
        map_success = False
        if scorer.answer_generator != 'self':
            existing_ans_dict = None
            answer_dir = os.path.dirname(answer_path)
            answer_file = os.path.basename(answer_path)
            available_subs = []
            for s in ['self','gpt3.5','gpt4']:
                if s in answer_file:
                    remaining_subs = [x for x in ['self','gpt3.5','gpt4'] if x != s]
                    available_subs.extend([deepcopy(answer_file).replace(s,x) for x in remaining_subs])
                    break
            for existing_ans_file in os.listdir(answer_dir):
                if any([p == existing_ans_file for p in available_subs]):
                    print (f'Exisiting ans dict found: {existing_ans_file}, Mapping raw sampled answer now')
                    existing_ans_dict = pickle.load(open(os.path.join(answer_dir,existing_ans_file),'rb'))
                    try:
                        ans_dict = map_list_of_dicts(existing_ans_dict,scored_qns,compare_key = 'instruction',map_key = 'raw_answer_sample')
                        map_success = True
                    except Exception as e:
                        print (e)
                        map_success = False
                    break 
                
        if not map_success:
            sample_ans_fn = partial(generate_ans,type_='raw_answer_sample')
            ans_dict = filter_none(async_call_fn(sample_ans_fn,scored_qns,max_workers,msg = 'Generating sample raw answers'))
            with open(answer_path,'wb') as f:
                pickle.dump(ans_dict,f)
    else:
        with open(answer_path,'rb') as f:
            ans_dict = pickle.load(f)
    
    ####################
    ## GENERATE SCORE ##
    ####################    
    # if score_key not in ans_dict[0]: 
    all_scored_instances = get_scored_ds(ans_dict,'raw_answer_sample')
    with open(answer_path,'wb') as f: 
        pickle.dump(all_scored_instances,f)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082,help = 'port for TGI')
    parser.add_argument("--config_path", type=str, default="",required=True)
    ## Dataset curation ##
    parser.add_argument("--answer_generator", type=str, default="self",required=False)
    parser.add_argument("--scoring_method", type=str, default="SelCheckGPT",required=False)
    parser.add_argument("--topic_generator", type=str, default="wiki",required=False)
    parser.add_argument("--use_tgi", action='store_true',help = 'use TGI for loaded model to do eval')
    parser.add_argument("--max_response_tokens", type=int, default=256,help = 'max tokens for answer generation')
    parser.add_argument("--num_topics",  type = int,default = 200,help = 'total qns to generate')
    parser.add_argument("--num_samples", type=int, default=10,help = 'number of sampled responses')
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--test_size",  type = int,default = 10)
    parser.add_argument("--questions_per_topic",  type = int,default = 10)
    parser.add_argument("--beta",  type = float,default = 1.0,help = 'Trade-off between observed and self-reflected confidence, for BSDetector only.')
    parser.add_argument("--question_filtering",  action = 'store_true',help= 'to filter question based on hallucination score.')
    parser.add_argument("--generate_sft",  action = 'store_true',help= 'generate only sft data')
    parser.add_argument("--openai_api_key_path",  type = str,default = 'openai_api_key.txt',help = 'a text file for openai api key, required only if using factscorer.')
    parser.add_argument("--ref_as_chosen",  action='store_true',help = 'if context is given, always used context answer as chosen')
    parser.add_argument("--iter",  type = int, default = 0, help = 'current iteration of DPO')
    args = parser.parse_args()
    # args.num_concurrent_calls = np.floor(320/args.num_samples).astype(int).item() # set max request at 320.
    args.num_concurrent_calls = 10
    ## Seed ## 
    seed_all(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    ## Get config ##
    with open(args.config_path,'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
    
    ds_config_path = f'configs/data/wiki.yaml'
    with open(ds_config_path,'r') as f:
        ds_config = yaml.safe_load(f)

    
    if not args.ref_as_chosen:
        assert not args.question_filtering, "Cannot use question filtering without ref_as_chosen."

    ## Test path ##
    ds_config['test_dataset_path'] = ds_config['test_dataset_path'].format(dataset_name = args.topic_generator)
    # if args.answer_generator in ['gpt4','gpt3.5'] or not os.path.exists(ds_config['test_dataset_path']) or not os.path.exists(config.question_path) or not os.path.exists(config.context_answer_path):
    assert args.openai_api_key_path != '','Need to provide openai api key for gpt4/gpt3.5'
    with open(args.openai_api_key_path,'r') as f:
        openai_api_key = f.read()
    if openai_api_key == '':
        raise ValueError('Need to provide openai api key for gpt4/gpt3.5')
    os.environ['OPENAI_API_KEY'] = openai_api_key
    
    ## DPO or SFT different mode
    question_path = config.question_path
    if not args.generate_sft:
        model_name = config.sft_model_path.format(answer_generator='self') 
        answer_path = config.dpo_answer_path.format(answer_generator=args.answer_generator)
        if args.question_filtering:
            model_name += '_qf'
            answer_path= answer_path.replace('.pkl','_qf.pkl')
        context_answer_path = None
        if args.iter > 0:
            answer_path = answer_path.replace('.pkl',f'_iter{args.iter}.pkl')
    else:
        model_name = config.base_model_name
        answer_path = config.sft_answer_path.format(answer_generator=args.answer_generator)
        context_answer_path = config.context_answer_path
    
    if not args.ref_as_chosen:
        answer_path = answer_path.replace('.pkl','_wo_context.pkl')
        if not args.generate_sft:
            model_name += '_wo_context'

    ## create dirs
    required_dirs = [ds_config['test_dataset_path'],answer_path,question_path]
    for required_paths in required_dirs:
        base_dir_name = os.path.dirname(required_paths)
        os.makedirs(base_dir_name,exist_ok=True)
    
    config.topic_generator = args.topic_generator
    config.answer_generator = args.answer_generator
    config.scoring_method = args.scoring_method
    
    ## Generate kwargs
    gen_kwargs = {
                    'questions':{'max_new_tokens':ds_config.get('max_question_tokens',128), 'do_sample':True,'temperature':0.5},
                    'gold_answer':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True,'details':True,'repetition_penalty':1.1,'temperature':0.3},
                    'raw_answer_sample':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True, 'temperature':config.temperature,'best_of':args.num_samples,'details':True,'top_p':0.9,'repetition_penalty':1.1},
                    'gold_answer_sample':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True, 'temperature':0.7,'best_of':args.num_samples,'details':True,'top_p':0.9,'repetition_penalty':1.1},
                    }
    if not args.generate_sft: # no sampling for sft-ed model.
        gen_kwargs['gold_answer']['do_sample'] = False
        gen_kwargs['gold_answer'].pop('temperature')
    
    # client
    if args.use_tgi:
        client = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    else:
        client = load_hf_model(model_name,quantized=True)
    
    base_tokenizer = load_tokenizer(model_name,padding_side='left')
    
    if args.num_samples != 10: # TEMP
        answer_path = answer_path.replace('.pkl',f'_{args.num_samples}.pkl')
        if args.num_samples == 1:
            gen_kwargs['raw_answer_sample'] = {'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':False,'details':True}
    
    #####################
    ## GET WIKI TOPICS ##
    #####################
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

    # if os.path.exists(ds_config['test_dataset_path']): ## only use this part when generating out-of-distribution test dataset (different topics)
    #     with open(ds_config['test_dataset_path'],'r') as f:
    #         test_ds = [json.loads(l) for l in f]
    #         existing_test_topics = defaultdict(list)
    #         for d in test_ds:
    #             existing_test_topics[d['category']].append(d['topic'])
    # else:
    
    existing_test_topics = None
    all_topics,sup_documents,test_topics = get_wiki(args.num_topics,ds_config,num_test_topics= args.test_size,topic2docu=topic2docu,existing_test_topics = existing_test_topics)

    # SCORER
    scorer = NLIScorer(client,model_name,base_tokenizer,config.scoring_method,args.beta,max_response_tokens=ds_config.get('max_response_tokens',128),answer_generator=config.answer_generator,use_tgi = args.use_tgi,ref_as_chosen=args.ref_as_chosen)
    
    open_generate_qns(client,
                    scorer,
                    all_topics,
                    sup_documents,
                    args.num_concurrent_calls,
                    gen_kwargs,
                    base_tokenizer,
                    model_name,
                    question_path=question_path,
                    answer_path = answer_path,
                    context_answer_path = context_answer_path,
                    test_path = ds_config['test_dataset_path'],
                    test_topics = test_topics,
                    qn_per_topic=args.questions_per_topic,
                    use_tgi=args.use_tgi,
                    generate_sft= args.generate_sft,
                    ref_as_chosen = args.ref_as_chosen
                    )

if __name__ == "__main__":
    main()