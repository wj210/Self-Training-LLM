import os
import json
import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer,AutoModelForCausalLM
import concurrent.futures
from tqdm import tqdm
from scorer import NLIScorer
import argparse
import torch
import pickle
from sentence_transformers import SentenceTransformer
from self_learning_training import do_train,do_test
from torch.nn.utils.rnn import pad_sequence
import random
import wandb
from utils import load_hf_model,load_tokenizer,get_prompt_and_extract_template,get_nouns_and_embeddings,get_topic_embedding_space
from templates import format_question_generation
from collections import defaultdict
from time import time
from functools import partial

random.seed(42)
np.random.seed(42)

def open_generate_qns(client,scorer,num_iterations,max_workers,gen_kwargs,generate_topic=False,topics=None,filter_size = -1,prompt_fn_dict=None,question_path=None):
    
    ques_prompt_fn = prompt_fn_dict['prompt_fn']['question_gen']
    ans_prompt_fn = prompt_fn_dict['prompt_fn']['answer_gen']
    
    def tgi_generate(prompt,prompt_type = 'topic'):
        return client.text_generation(prompt, **gen_kwargs[prompt_type])
    
    def generate_qns(topic):
        if generate_topic:
            topic_prompt = "Propose a topic that you would like to learn more about. Answer with only the three proposed topics concisely without elaboration.\n One such example is:\n1. Elementary mathematics\n2. Pyschology\n3. Quantum mechanics\n\nTopics:"
            topic =  tgi_generate(ques_prompt_fn(topic_prompt),'topic')
            
        qn_prompt = format_question_generation(topic)
        question =  tgi_generate(ques_prompt_fn(qn_prompt),'questions')
        return {'topic':topic,'question':question}
    
    def generate_ans(qns_dict):
        qns = qns_dict['question']
        qn_prompt = ans_prompt_fn(qns)
        ref_ans = tgi_generate(qn_prompt,'ref_answer')
        sample_ans = tgi_generate(qn_prompt,'sample_answers')
        return {'ref_ans':ref_ans,'sample_ans':sample_ans,'instruction':qns,'topic':qns_dict['topic']} # each is a dict with details (logprobs and text)

    if generate_topic:
        assert topics == None
        topics = list(range(num_iterations))
        
    # Generate qns and answer dict, note that sampled/ref answers are shared across all answer_generator types, only difference is how we chose the preference response for DPO #
    
    if not os.path.exists(question_path):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            qns_dict = list(tqdm(executor.map(generate_qns,topics),total=len(topics),desc='Generating questions'))

        ## Get answers ##
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            ans_dict = list(tqdm(executor.map(generate_ans,qns_dict),total=len(qns_dict),desc='Generating answers'))
        
        ## Get score and filter questions and generate DPO dataset ##
        all_qn_confidences = []
        if scorer:
            for ans_d in tqdm(ans_dict,total = len(ans_dict),desc='Scoring questions based on confidence'):
                ref_ans = ans_d['ref_ans']
                sample_ans = ans_d['sample_ans']
                instruction = ans_d['instruction']
                score_dict = scorer.get_score(instruction,ref_ans,sample_ans)
                score_dict = {**score_dict,'topic':ans_d['topic']}
                all_qn_confidences.append(score_dict)
            
            if filter_size >0:
                if scorer.scoring_method == 'BSDetector':
                    filter_size = int(filter_size*len(all_qn_confidences))
                    all_qn_confidences = sorted(all_qn_confidences,key = lambda x: x['confidence'])[:filter_size] # get lowest confidence questions
                elif scorer.scoring_method == 'semantic_consistency':
                    all_qn_confidences = [x for x in all_qn_confidences if len(list(x['semantic_clusters'].keys())) >1] # remove questions with only 1 cluster
                    filter_size = int(filter_size*len(all_qn_confidences))
                    all_qn_confidences = sorted(all_qn_confidences,key = lambda x: x['overall_entropy'],reverse = True)[:filter_size] # get highest entropy qns.
            
            ## Save the set of questions along with sampled answer,ref_answer and other scores required to generate chosen dataset.##
            with open(question_path,'wb') as f:
                pickle.dump(all_qn_confidences,f)
        else:
            raise ValueError('Scorer not provided')
    else:
        with open(question_path,'rb') as f:
            all_qn_confidences = pickle.load(f)

    ## Get DPO dataset , get chosen answer based on google search, LLM generator or heuristics (confidence/entropy) ##
    if scorer.answer_generator in ['gpt4','gpt3.5']: # leverage multithreading, since making api calls
        max_workers = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor: # set max to 10, to prevent google blocking
            generated_ds = list(tqdm(executor.map(scorer.get_dpo_sample,all_qn_confidences),total=len(all_qn_confidences),desc='Generating dpo samples'))
    else:
        generated_ds = [scorer.get_dpo_sample(qn) for qn in all_qn_confidences]            

    pre_ds_len = len(generated_ds)
    generated_ds = [t for t in generated_ds if t is not None]
    print (f'Number of questions before filtering: {pre_ds_len}, after filtering: {len(generated_ds)}')
    
    return generated_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--model_name", type=str, default="Intel/neural-chat-7b-v3-3")
    parser.add_argument("--scoring_method", type=str, default='BSDetector')
    parser.add_argument("--topic_generator", type=str, default='oracle',choices = ['oracle','self'], help = 'oracle uses wordnet to source for possible topics and select lowest probable, self uses the model to self generate topics')
    parser.add_argument("--answer_generator", type=str, default='oracle', help = 'oracle uses google to search for documents for the model to reference, self sets answer to highest confidence answer from model, gpt3.5/4 uses OPENAI')
    parser.add_argument("--use_tgi", type=bool, default=False,help = 'use TGI for loaded model to do eval')
    parser.add_argument("--max_response_tokens", type=int, default=256,help = 'max tokens for answer generation')
    parser.add_argument("--training", type = bool,default = False,help = 'perform training')
    parser.add_argument("--testing",  type = bool,default = False,help = 'perform testing')
    parser.add_argument("--peft_path",  type = str,default = 'uncertainty/configs/lora.yaml',help = 'get lora parameters')
    parser.add_argument("--use_peft",  type = bool,default = False,help = 'use peft')
    parser.add_argument("--num_iterations",  type = int,default = 200,help = 'total qns to generate')
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--topic_ratio",  type = float,default = 1.0,help = 'topic/question ratio, if 0.5 means 2 questions per topic, max 1.0, min > 0.')
    parser.add_argument("--beta",  type = float,default = 0.7,help = 'Trade-off between observed and self-reflected confidence, for BSDetector only.')
    args = parser.parse_args()
    
    ## Wandb settings ##
    # wandb.login()
    os.environ["WANDB_PROJECT"]="self_learning_training"
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    
    ## Dataset path ##
    data_path = 'data'
    os.makedirs(data_path,exist_ok=True)
    model_last_name = args.model_name.split('/')[-1]
    os.makedirs(f'data/train',exist_ok=True)
    os.makedirs(f'data/test',exist_ok=True)
    scoring_name = args.scoring_method
    if scoring_name in 'BSDetector' and args.beta == 1.0:
        scoring_name += '_pureNLI'
    
    train_dataset_path = f'data/train/{model_last_name}_{args.answer_generator}_{scoring_name}.jsonl'
    test_dataset_path = f'data/test/{model_last_name}_{args.answer_generator}_{scoring_name}.jsonl'
    
    ## topic path ##
    """
    Check through existing topic generated, and see if we need to generate more topics according to total topics required
    Load exisiting topics if available and generate until we have the required number of topics.
    """
    assert args.topic_ratio <= 1.0 and args.topic_ratio > 0, 'topic ratio must be between 0 and 1'
    total_topics = int(args.num_iterations*args.topic_ratio)
    topic_path = f'data/topics_{total_topics}.pkl'
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
        total_topics_to_generate = max(int((total_topics - len(prev_topics))*args.topic_ratio),0)
    else: # if no topics file, then we generate all
        total_topics_to_generate = total_topics
    
    ## Question ##
    question_dir = 'data/questions'
    os.makedirs(question_dir,exist_ok=True)
    question_path = os.path.join(question_dir,f'{scoring_name}.pkl')

    ## Model path ##
    model_dir = 'model_checkpoints'
    os.makedirs(model_dir,exist_ok=True)
    model_path = f'{model_dir}/{model_last_name}_{args.answer_generator}_{scoring_name}'
    
    ## Result path ##
    result_dir = 'test_results'
    os.makedirs(result_dir,exist_ok=True)
    result_path = f'{result_dir}/{model_last_name}_{args.answer_generator}_{scoring_name}.txt'
    
    ## Generate kwargs ##
    gen_kwargs = {'topic':{'max_new_tokens':64, 'do_sample':True, 'temperature':1.0},
                    'questions':{'max_new_tokens':64, 'do_sample':True, 'temperature':1.0},
                    'ref_answer':{'max_new_tokens':args.max_response_tokens, 'do_sample':False, 'repetition_penalty':1.1,'details':True},
                    'sample_answers':{'max_new_tokens':args.max_response_tokens, 'do_sample':True, 'temperature':1.0,'best_of':10,'repetition_penalty':1.1,'details':True}}
    
    base_tokenizer = load_tokenizer(args.model_name,padding_side='left') # autoregressive model uses left padding
    prompt_fn_dict = get_prompt_and_extract_template(args.model_name)
    
    if not os.path.exists(train_dataset_path) or not os.path.exists(test_dataset_path):
        ###################
        ## Generate data ##
        ###################
        client = InferenceClient(model = f"http://127.0.0.1:{args.port}")
        num_workers = 16 # parallel API calls to TGI
        
        if args.topic_generator == 'oracle':
            if total_topics_to_generate > 0:
                """
                Generate topics using wordnet and select most uncertain topics based on model logprobs.
                """
                nearest_neighour_topics = 20
                ## From self_learning_utils.py ##
                embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
                base_model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=torch.bfloat16,trust_remote_code=True).cuda()
                topic_space = get_nouns_and_embeddings(embedder)
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
                
        else:
            prev_topics = None
        
        if prev_topics is not None and isinstance(prev_topics,set):
            prev_topics = list(prev_topics)
        if prev_topics is not None and args.topic_ratio < 1.0: # create duplicate topics if topic ratio < 0.1
            num_dup_per_topic = int(1/args.topic_ratio)
            all_topics = sum([[x]*num_dup_per_topic for x in prev_topics],[])
        else:
            all_topics = prev_topics

        scorer = NLIScorer(client,"microsoft/deberta-large-mnli",base_tokenizer,args.scoring_method,args.beta,max_response_tokens=args.max_response_tokens,answer_generator=args.answer_generator,prompt_fn_dict=prompt_fn_dict)
        ## Generate qns with score ##
        all_scored_instances = open_generate_qns(client,scorer,args.num_iterations,num_workers,gen_kwargs,generate_topic = True if args.topic_generator == 'self' else False,topics = all_topics,filter_size = args.filter_size,prompt_fn_dict = prompt_fn_dict,question_path=question_path)

        # Save individual answer mode dataset # (For eval later on)
        test_size = 0.2
        test_size = min(int(test_size*len(all_scored_instances)),500) # we limit test size to 500
        random.shuffle(all_scored_instances)
        train_ds = all_scored_instances[:-test_size]
        test_ds = all_scored_instances[-test_size:]
            
        with open(train_dataset_path,'w') as f:
            for instance in train_ds:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')
        
        with open(test_dataset_path,'w') as f:
            for instance in test_ds:
                json.dump(instance,f,ensure_ascii=False)
                f.write('\n')
                
    ##############
    ## Training ##
    ##############
    
    if args.training:
        with open(train_dataset_path,'r') as f:
            train_ds = [json.loads(line) for line in f]
        print ('**Training model**')
        do_train(
            ds=train_ds,
            questions_with_hallucination = None,
            prompt_format_fn = prompt_fn_dict['prompt_fn']['answer_gen'],
            extract_response_fn=None,
            tokenizer = base_tokenizer,
            model_name_or_path=args.model_name,
            batch_size=4,
            max_epochs=10,
            lr=3e-5,
            deterministic=True,
            saved_model_path=model_path,
            max_response_len=args.max_response_tokens,
            use_peft = args.use_peft,
            peft_path = args.peft_path,
        )
        
        # if args.use_tgi and args.use_peft:
        #     """
        #     If we use TGI and we are using peft, we need to reload the model in 16bit and merge the adapter layers.
        #     Else, we can just load it as it is and merge it everytime we do inference with regular model inference.
        #     """
        #     trained_model = load_hf_model(args.model_name,model_path,use_tgi=True)
        #     model_saved_name ='neuralchat_self-confidence' if args.scoring_method == 'BSDetector' else 'neuralchat_self-entropy'
        #     trained_model.push_to_hub(model_saved_name)
        #     base_tokenizer.push_to_hub(model_saved_name)
        #     exit('load new model from tgi before running again')

    #############
    ## Testing ##
    #############
    """
    Evaluate on Test set, get confidence score and for pre and post-training.
    Generate questions and get overall_confidence score for dataset. (may or maynot decrease). Even though we ask the model to give us the lowest confidence questions, it may still be more confident as the model is improved.
    """
    
    if args.testing:
        
        with open(test_dataset_path,'r') as f:
            test_ds = [json.loads(line) for line in f]
        with open(train_dataset_path,'r') as f:
            train_ds = [json.loads(line) for line in f]
        ## Load trained model ##
        if args.use_tgi:
            eval_model = InferenceClient(model = f"http://127.0.0.1:{args.port}")
        else:
            eval_model = load_hf_model(args.model_name,model_path,use_tgi=False)
        
        scorer = NLIScorer(eval_model,"microsoft/deberta-large-mnli",base_tokenizer,args.scoring_method,args.beta,max_response_tokens=args.max_response_tokens,answer_generator = args.answer_generator,use_tgi = args.use_tgi,prompt_fn_dict=prompt_fn_dict)
        
        test_keys = ['answer_confidence','question_confidence','answer_performance']
        
        if 'question_confidence' in test_keys:
            ## Pre training qn confidence, for question confidence, we compare with previously generated training set ##
            pre_generated_topics = [x['topic'] for x in train_ds]
            if scorer.scoring_method == 'BSDetector':
                pre_question_score = np.mean([x['qn_confidence'] for x in train_ds])
                pre_answer_score = [x['qn_confidence'] for x in test_ds]
            elif scorer.scoring_method == 'semantic_consistency':
                pre_question_score = np.mean([x['qn_entropy'] for x in train_ds])
                pre_answer_score = [x['qn_entropy'] for x in test_ds]
            
            extra_kwargs = {
                            'pre_question_score':pre_question_score,
                            'pre_answer_score':pre_answer_score, # pre-confidence is computed prior when generating test dataset
                            'topics':pre_generated_topics,
                            'use_tgi':args.use_tgi}
            
        print ('**Testing model**')
        test_results = do_test(ds=test_ds,
                prompt_fn_dict=prompt_fn_dict,
                model=eval_model,
                tokenizer=base_tokenizer,
                scorer=scorer,
                batch_size = 4,
                gen_kwargs=gen_kwargs,
                out_file=result_path,
                test_keys = test_keys,
                **extra_kwargs if 'question_confidence' in test_keys else {})

if __name__ == "__main__":
    main()