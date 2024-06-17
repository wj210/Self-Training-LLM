import os
import json
import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
from scorer import NLIScorer,async_process
import argparse
import torch
import pickle
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
import random
from utils import *
from data_utils import *
from topic_generator import get_topic_embedding_space,get_wordnet_nouns_and_embeddings
from templates import format_question_generation
from functools import partial
from copy import deepcopy
from dataclasses import asdict
import yaml
from rouge_score import rouge_scorer
from types import SimpleNamespace

def open_generate_qns(client,scorer,num_iterations,max_workers,gen_kwargs,tokenizer,model_name,topics=None,questions=None,question_path=None,exclude_questions = None,few_shots = None,num_fs=0,kwargs = {}):
    """
    client is the inference client for TGI (assume that we use this.)
    scorer is the scorer object that we use to score questions and answers.
    num_iterations is the number of topics to generate. (only used if self-generated topics)
    max_workers = concurrent api calls
    topics = wordnet or fixed ds topics
    known_question_path/unknown to save QH and Q_notH
    exclude_questions = list of QA_sample from data_utils.py
    """
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score_key = SCORE_KEY[scorer.scoring_method]
    
    def tgi_generate(prompt,prompt_type = 'topic'):
        return client.text_generation(prompt, **gen_kwargs[prompt_type])
    
    def generate_qns(topic,few_shot=None):
        if not isinstance(topic,str): # is a not string, then flag for self-generating topic
            topic_prompt = "Propose a topic that you would like to learn more about. Answer with only the three proposed topics concisely without elaboration.\n One such example is:\n1. Elementary mathematics\n2. Pyschology\n3. Quantum mechanics\n\nTopics:"
            topic =  tgi_generate(format_response(topic_prompt,model_name,tokenizer,mode='question'),'topic')

        ## Few shot questions ##
        if few_shot is not None:
            few_shot = few_shot[topic][:num_fs]
            random.shuffle(few_shot)
            few_shot_qns = [[{'role':'user','content':format_question_generation(fs.topic)},
                            {'role':'assistant','content':fs.instruction}] for fs in few_shot]
            few_shot_qns = sum(few_shot_qns,[]) #flatten
            qn_prompt = few_shot_qns + [{'role':'user','content':format_question_generation(topic)}]
        else:
            qn_prompt = [{'role':'user','content':format_question_generation(topic)}]
            
        if is_instruct_tuned:
            qn_prompt = format_response(qn_prompt,model_name,tokenizer,mode='question')
        else:
            for qn_p in qn_prompt:
                if qn_p['role'] == 'assistant':
                    qn_p['content'] = f"\nQ: {qn_p['content']}"
            qn_prompt[-1]['content'] += '\nQ: '
            qn_prompt = "\n\n".join([x['content'] for x in qn_prompt])

        question =  tgi_generate(qn_prompt,'questions')
        if not is_instruct_tuned:
            question = question.split('\n\n')[0].strip()
        if kwargs.get('single_qn',False): # only for truthfulqa, we generate single question, take only the first question.
            question = question.split('?')[0].strip() + '?'
        return {'topic':topic,'instruction':question}
    
    def generate_ans(qns_dict,few_shot=None):
        qns = qns_dict['instruction']
        topic = qns_dict['topic']
        if few_shot is not None:
            few_shot = format_fs_qa(few_shot,is_instruct_tuned)
        else:
            few_shot = []
        if is_instruct_tuned:
            ans_prompt = few_shot + [{'role':'user','content':qns}]
            formatted_ans_prompt = format_response(ans_prompt,model_name,tokenizer,mode='answer')
        else:
            formatted_ans_prompt = "\n\n".join(few_shot + [f"Q: {qns}\nA: "])
        ref_ans = tgi_generate(formatted_ans_prompt,'ref_answer')
        sample_answer = tgi_generate(formatted_ans_prompt,'sample_answers')
        
        out = {'ref_answer':ref_ans,'sample_ans':sample_answer,'instruction':qns,'topic':topic}
        return out 
    
    def get_scored_ds(ans_dict):
        all_qn_confidences = []
        for ans_d in tqdm(ans_dict,total = len(ans_dict),desc=f'Scoring questions based on {scorer.scoring_method}'):
            if 'ref_ans' in ans_d:
                ref_ans  = ans_d['ref_ans']
            else:
                ref_ans = ans_d['ref_answer']
            sample_ans = ans_d['sample_ans']
            instruction = ans_d['instruction']
            score_dict = scorer.get_score(instruction,ref_ans,sample_ans)
            score_dict = {**ans_d,**score_dict}
            all_qn_confidences.append(score_dict)
        return all_qn_confidences

    if topics == None:
        topics = list(range(num_iterations))
        
    # Generate qns and answer dict, note that sampled/ref answers are shared across all answer_generator types, only difference is how we chose the preference response for DPO #
    if not os.path.exists(question_path):
        if questions is None:
            ## If want to include newly generated questions as few-shot, uncomment this ##
            # existing_qns = deepcopy(few_shots)
            # final_qns_dict = []
            # for topic in tqdm(topics,'Generating questions'):
            #     topic_fs = existing_qns.get(topic,[])
            #     random.shuffle(topic_fs)
            #     new_qn_dict = generate_qns(topic,topic_fs[:num_fs])
            #     if len(topic_fs) > 0 and not is_duplicate(r_scorer,new_qn_dict['instruction'],[t.instruction for t in topic_fs]): # use ROUGE-L to check for similarity with existing qn.
            #         final_qns_dict.append(new_qn_dict)
            #         existing_qns[topic].append(SimpleNamespace(**new_qn_dict)) # to re-sample few-shot for diversity.  
            # if exclude_questions is not None: # ensure no duplicate from test dataset.
            #     final_qns_dict = [x for x in final_qns_dict if x['instruction'] not in set([d.instruction for d in exclude_questions])]  
            # print ('Remove {} duplicate questions'.format(len(topics) - len(final_qns_dict)))
            
            generate_qns_fn = partial(generate_qns,few_shot=few_shots)
            qns_dict = async_process(generate_qns_fn,topics,max_workers*2,msg = 'Generating questions')
            
            ## Remove duplicate questions ## 
            non_duplicate_qn_dicts = [qns_dict[0]]
            topic_qn_dicts = defaultdict(list) # just to make checking for similar qns in each topic faster
            topic_qn_dicts[qns_dict[0]['topic']].append(qns_dict[0]['instruction'])
            remaining_qn_dicts =[]
            for q in qns_dict[1:]: # ensure that we start off with at most 1 qn per topic
                if q['topic'] not in set([eq['topic'] for eq in non_duplicate_qn_dicts]):
                    non_duplicate_qn_dicts.append(q) 
                    topic_qn_dicts[q['topic']].append(q['instruction'])
                else:
                    remaining_qn_dicts.append(q)
            for q in tqdm(remaining_qn_dicts, total =len(remaining_qn_dicts),desc = 'Checking for similar questions based on rouge'): # check based on topic
                if not is_duplicate(r_scorer,q['instruction'],topic_qn_dicts[q['topic']]):
                    non_duplicate_qn_dicts.append(q)
                    topic_qn_dicts[q['topic']].append(q['instruction'])
            del topic_qn_dicts
            del remaining_qn_dicts
            print ('Remove {} duplicate questions'.format(len(qns_dict) - len(non_duplicate_qn_dicts)))
            
            ## Remove any qns that is passed in ##
            if exclude_questions is not None:
                final_qns_dict = [x for x in non_duplicate_qn_dicts if x['instruction'] not in set([d.instruction for d in exclude_questions])]
        else:
            final_qns_dict = questions # already predefined.
        
        if scorer.answer_generator != 'oracle_answer':
            ## get answers ##
            generate_ans_fn = partial(generate_ans,few_shot=kwargs.get('answer_few_shot',None)) # Set few shot to be None.
            ans_dict = async_process(generate_ans_fn,final_qns_dict,max_workers,msg = 'Generating answers')
            with open(question_path,'wb') as f:
                pickle.dump(ans_dict,f)
        else:
            ans_dict = final_qns_dict
            
    if scorer.answer_generator != 'oracle_answer':
        with open(question_path,'rb') as f:
            ans_dict = pickle.load(f)
        
        if score_key not in ans_dict[0]: # not yet scored.
            ans_dict = get_scored_ds(ans_dict)
            with open(question_path,'wb') as f: # re-update the question_set with other approach scores.
                pickle.dump(ans_dict,f)
    
        unknown_qns = return_question_type(ans_dict,scorer.scoring_method) ## TEMP
        # unknown_qns = ans_dict
        ## Get DPO dataset , get chosen answer based on google search, LLM generator or heuristics (confidence/entropy) ##
        get_ds_fn = partial(scorer.get_dpo_sample,few_shots=few_shots)
        generated_ds = []
        for unknwn_qn in tqdm(unknown_qns,desc='Generating dpo samples',total = len(unknown_qns)):
            try:
                generated_ds.append(get_ds_fn(unknwn_qn))
            except Exception as e:
                print (e)
        generated_ds = [t for t in generated_ds if t is not None]
    else:
        generated_ds = ans_dict
        for g in generated_ds: # replace key
            g['chosen_ans'] = g.pop('answer')
            g['rejected_ans'] = random.choice(g['incorrect_answer'])
        # else:
        #     for g in generated_ds: # replace key
        #         g['chosen_ans'] = g.pop('actual_label')
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
    parser.add_argument("--num_iterations",  type = int,default = 200,help = 'total qns to generate')
    parser.add_argument("--num_concurrent_calls", type=int, default=32,help = 'max_api_calls to TGI at a time')
    parser.add_argument("--num_samples", type=int, default=5,help = 'number of sampled responses')
    parser.add_argument("--questions_per_topic",  type = int,default = 10)
    parser.add_argument("--beta",  type = float,default = 0.7,help = 'Trade-off between observed and self-reflected confidence, for BSDetector only.')
    parser.add_argument("--answer_generator_port", type=int, default=8083,help = 'port for TGI to generate answer, only used for mistral_8x7 if loaded locally.')
    parser.add_argument("--test_question_per_topic",  type = int,default =-1,help = 'if more than -1, we only test on this number of questions per topic')
    args = parser.parse_args()
    assert args.num_concurrent_calls*args.num_samples <= 320, 'Keep max api calls to <320 to prevent crash.'
    ## Seed ## 
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
    ## Get config ##
    with open(args.config_path,'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
    
    if 'fixed' in args.topic_generator or 'predefined' in args.topic_generator or 'oracle' in args.topic_generator: # using known dataset
        ds_name = '_'.join(args.topic_generator.split('_')[1:]) # name of ds
        ds_config_path = f'configs/data/{ds_name}.yaml'
        with open(ds_config_path,'r') as f:
            ds_config = yaml.safe_load(f)
    else:
        ds_name = 'self-generated'
        ds_config = {}
    
    ## Make dir ##
    ## Train/Test Dir ##
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
    config.fs_path = config.fs_path.format(dataset_name = ds_name)
    
    if args.topic_generator in ['self','wordnet']:
        test_d_path = 'self_' + ds_config['test_dataset_path'].split('/')[-1] 
        base_test_dir = os.path.dirname(ds_config['test_dataset_path'])
        ds_config['test_dataset_path'] = base_test_dir + '/' + test_d_path
    
    ## Question path ##
    config.question_path = config.question_path.format(topic_generator=args.topic_generator)
    ## create dirs
    for k in config.__dict__.keys():
        if 'path' in k:
            base_dir_name = os.path.dirname(config.__dict__[k])
            os.makedirs(base_dir_name,exist_ok=True)
    
    ## End make dir ##
    config.topic_generator = args.topic_generator
    config.answer_generator = args.answer_generator
    config.scoring_method = args.scoring_method
    
    
    ## Generate kwargs ##
    gen_kwargs = {'topic':{'max_new_tokens':5, 'do_sample':True, 'temperature':1.0,'repetition_penalty':1.1},
                    'questions':{'max_new_tokens':ds_config.get('max_question_tokens',128), 'do_sample':True, 'temperature':0.5,'repetition_penalty':1.1},
                    'ref_answer':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':False, 'repetition_penalty':1.1,'details':True},
                    'sample_answers':{'max_new_tokens':ds_config.get('max_response_tokens',128), 'do_sample':True, 'temperature':1.0,'best_of':args.num_samples,'repetition_penalty':1.1,'details':True}}
    
    base_tokenizer = load_tokenizer(config.model_name,padding_side='left') # autoregressive model uses left padding
    ###################
    ## Generate data ##
    ###################
    """
    Check through existing topic generated, and see if we need to generate more topics according to total topics required
    Load exisiting topics if available and generate until we have the required number of topics.
    """
    #######################
    ## topic for wordnet ##
    #######################
    assert args.questions_per_topic >= 1, 'number of questions per topic must be > 1'
    if config.topic_generator == 'wordnet': # wordnet has alot of topics, and we sample them.
        total_topics = args.num_iterations//args.questions_per_topic
        topic_path = f'data/{args.topic_generator}/topics_{total_topics}.pkl'
        current_data_files = os.listdir('data')
        max_available_topics,max_file_path = 0,''
        prev_topics = set()
        for f in current_data_files:
            if 'topics' in f:
                curr_num = int(f.split('.pkl')[0].split('_')[-1])
                if curr_num > max_available_topics:
                    max_available_topics = curr_num
                    max_file_path = os.path.join('data',f)
                else:
                    os.remove(os.path.join('data',f)) # if lesser, just remove that file, note that all topic files have to have the postfix of the number of topics.
        if max_file_path != '': # load existing topics
            prev_topics = pickle.load(open(max_file_path,'rb'))
            prev_topics = set(prev_topics)
            total_topics_to_generate = max(int(total_topics - len(prev_topics)),0)
        else: # if no topics file, then we generate all
            total_topics_to_generate = total_topics
        
        print ('Total topics to generate:',total_topics_to_generate)
    else:
        total_topics_to_generate = 0 # others either self-generate or use a pre-fixed set of topics like MMLU

    client = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    
    exclude_instructions = [] # for fixed ds, where we dont not want the training set to include test qn
    fs_shot_generation = []
    defined_ds = None # for predefined ds, only need is predefined question set
    qn_generation_kwargs = {}
    if config.topic_generator == 'wordnet':
        if total_topics_to_generate > 0:
            """
            Generate topics using wordnet and select most uncertain topics based on model logprobs.
            """
            nearest_neighour_topics = 20
            ## From self_learning_utils.py ##
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
            base_model = AutoModelForCausalLM.from_pretrained(config.model_name,torch_dtype=torch.bfloat16,trust_remote_code=True).cuda()
            topic_space = get_wordnet_nouns_and_embeddings(embedder)
            all_embeddings = topic_space['all_embeddings']
            all_topics = topic_space['all_nouns']
            topic_embedding_space = get_topic_embedding_space(all_embeddings=all_embeddings)
            print ('Number of available topics:',len(all_topics))
            
            starting_prompt = 'Name a particular topic that you are confident in.\nTopic:'
            tokenized_start = base_tokenizer.encode(starting_prompt, add_special_tokens=False)
            len_start = len(tokenized_start)
            
            for _ in tqdm(range(total_topics_to_generate),total = total_topics_to_generate,desc=f'Self sampling {total_topics_to_generate} topics from wordnet'): # sample topics from wordnet
                initial_point = torch.randn([384], requires_grad=False, device=torch.device('cpu'))
                _, emb_indices = topic_embedding_space.query(initial_point.cpu(), k=nearest_neighour_topics) 
                topic_candidates = [all_topics[idx].split(" ### ")[0] for idx in emb_indices]
                topic_candidates = [x.replace("_", " ") for x in topic_candidates]
                topic_candidates = [x for x in topic_candidates if x not in prev_topics]
                if len(topic_candidates) == 0:
                    continue
                ## NEW: We select the least confident out of the sampled nearest topic ## 
                tokenized_whole_context = [base_tokenizer.encode(starting_prompt+ t, add_special_tokens=False) for t in topic_candidates]
                tokenized_labels = [t[len_start:] for t in tokenized_whole_context]
                tokenized_input = [tokenized_start + l for l in tokenized_labels]
                tokenized_input = [t[:-1] for t in tokenized_input] # remove the last token since auto-regressive
                unpadded_len = [len(t) for t in tokenized_input] # record down unpadded len
                tokenized_topics = pad_sequence([torch.tensor(t,dtype=torch.long) for t in tokenized_input],batch_first=True,padding_value=0).cuda()
                logprobs = torch.nn.functional.log_softmax(base_model(tokenized_topics).logits,dim=-1).detach().cpu()
                all_logprobs = []
                for ul,lp,lab in zip(unpadded_len,logprobs,tokenized_labels):
                    lp = lp[:ul]
                    seq_lp = lp[-len(lab):]
                    selected_lp = torch.gather(seq_lp,1,torch.tensor(lab,dtype=torch.long).unsqueeze(-1)).squeeze(-1) # lp is LXD, lab is L
                    sum_lp = torch.sum(selected_lp).numpy()
                    all_logprobs.append(sum_lp)
                min_lp_idx = np.argmin(all_logprobs)
                prev_topics.add(topic_candidates[min_lp_idx])
            with open(topic_path,'wb') as f:
                pickle.dump(list(prev_topics),f)
                
    elif 'fixed' in config.topic_generator or 'predefined' in config.topic_generator or 'oracle' in config.topic_generator: # TODO add more fixed topic ds
        prev_topics = get_fixed_topics(ds_name)
        test_ds,fs_shot_generation,defined_ds = get_fixed_ds(ds_config,args.questions_per_topic,args.test_question_per_topic,args.topic_generator)
        exclude_instructions = deepcopy(test_ds)
        defined_ds = [vars(d) for d in defined_ds]
        if 'fixed' in config.topic_generator:
            defined_ds = None # we dont need this for fixed ds since question is self-generated.
            
        if 'truthful_qa' in config.topic_generator:
            qn_generation_kwargs['answer_few_shot'] = [SimpleNamespace(**d) for d in ds_config['test_few_shot']]
            qn_generation_kwargs['single_qn'] = True
        ## Few shot generation
        if not os.path.exists(config.fs_path):
            saved_fs = deepcopy(fs_shot_generation)
            with open(config.fs_path,'wb') as f:
                for topic,fs_shots in saved_fs.items():
                    saved_fs[topic] = [asdict(fs) for fs in fs_shots]
                pickle.dump(saved_fs,f)
        else:
            with open(config.fs_path,'rb') as f:
                fs_shot_generation = pickle.load(f)
                fs_shot_generation = {k:[SimpleNamespace(**fs) for fs in v] for k,v in fs_shot_generation.items()}
                
    elif config.topic_generator == 'self':
        prev_topics = None
    else:
        raise ValueError('Invalid topic generator')
    
    if prev_topics is not None:
        if isinstance(prev_topics,set):
            prev_topics = list(prev_topics)
        if args.questions_per_topic > 1.0: # create duplicate topics if topic ratio < 0.1
            prev_topics = sum([[x]*args.questions_per_topic for x in prev_topics],[])
    
    ###################
    ## QA generation ##
    ###################
    # scorer
    scorer = NLIScorer(client,config.model_name,base_tokenizer,config.scoring_method,args.beta,max_response_tokens=ds_config.get('max_response_tokens',128),answer_generator=config.answer_generator,answer_generator_port=args.answer_generator_port)
    
    ## Generate qns with score ##
    if isinstance(fs_shot_generation,list) and len(fs_shot_generation) == 0:
        fs_shot_generation = None
    
    all_scored_instances = open_generate_qns(client,scorer,args.num_iterations,args.num_concurrent_calls,gen_kwargs,base_tokenizer,config.model_name,
                                                topics = prev_topics,
                                                questions = defined_ds,
                                                question_path=config.question_path,
                                                exclude_questions=exclude_instructions,
                                                few_shots =fs_shot_generation,
                                                num_fs = ds_config.get('few_shot',0),
                                                kwargs = qn_generation_kwargs)

    # Save individual answer mode dataset # (For eval later on)
        
    if 'wordnet' in config.topic_generator: # if fixed ds, we only save train set, testset is loaded during testing.
        assert len(test_ds) == 0, 'Test set should be empty if not fixed ds.'
        test_size = 0.2
        test_size = min(int(test_size*len(all_scored_instances)),500) # we limit test size to 500
        random.shuffle(all_scored_instances)
        train_ds = all_scored_instances[:-test_size]
        test_ds = all_scored_instances[-test_size:]
    else:
        train_ds = all_scored_instances
    
    if not os.path.exists(ds_config['test_dataset_path']):
        with open(ds_config['test_dataset_path'],'w') as f:
            for instance in test_ds:
                json.dump(asdict(instance),f,ensure_ascii=False)
                f.write('\n')
            
    with open(config.train_dataset_path,'w') as f:
        for instance in train_ds:
            json.dump(instance,f,ensure_ascii=False)
            f.write('\n')
    
if __name__ == "__main__":
    main()