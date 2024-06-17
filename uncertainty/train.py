import os
import json
import argparse
import yaml
from types import SimpleNamespace
from self_learning_training import do_train
from utils import *
from data_utils import *
import pickle
from scorer import NLIScorer
from collections import Counter
from accelerate import Accelerator
from datasets import concatenate_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="",required=True)
    parser.add_argument("--answer_generator", type=str, default="self")
    parser.add_argument("--scoring_method", type=str, default="SelfCheckGPT")
    parser.add_argument("--topic_generator", type=str, default="wiki")
    parser.add_argument("--max_epochs",  type = int,default = -1)
    parser.add_argument("--max_steps",  type = int,default = -1)
    parser.add_argument("--training_batch_size",  type = int,default = 4)
    parser.add_argument("--peft_path",  type = str,default = 'configs/training/lora.yaml',help = 'get lora parameters')
    parser.add_argument("--use_peft",  action = 'store_true',help = 'use peft')
    parser.add_argument("--question_filtering",  action = 'store_true',help = 'use peft')
    parser.add_argument("--question_filtering_threshold", default = 0.5,type=float,help = 'threshold to get good quality samples')
    parser.add_argument("--multiple_pref",  action = 'store_true',help= 'Create multiple preference samples per instruction')
    parser.add_argument("--unknown_threshold",  default = 0.5,help= 'score to select the split unknown samples',type = float)
    parser.add_argument("--ref_as_chosen", action = 'store_true')
    parser.add_argument("--save_last_only",  action = 'store_true')
    parser.add_argument("--normalize_length",  action = 'store_true')
    parser.add_argument("--mode", default = 'sft',type = str,help = 'sft or dpo')
    parser.add_argument("--length_penalty",  default = 0.,type = float,help = "favor shorter chosen responses") # from https://arxiv.org/pdf/2403.19159.pdf
    parser.add_argument("--beta",  default= 0.3,type = float)
    parser.add_argument("--gradient_accumulation_steps",  default= 1,type = float)
    
    args = parser.parse_args()
    
    seed_all(42)
    
    assert (args.max_epochs > 0 and args.max_steps < 0) or (args.max_epochs < 0 and args.max_steps > 0), 'Either max_epochs or max_steps should be set.'
    assert not (args.length_penalty > 0 and args.normalize_length), 'Should not normalize length and use length penalty at the same time.'
    accelerator = Accelerator()
    is_main_process = accelerator.is_local_main_process
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    
    path_flag = ''
    
    if args.multiple_pref and args.question_filtering :
        assert args.question_filtering_threshold != 1.0, 'Remember to assign the threshold for golden samples'
    
    ## SFT ##
    if args.mode == 'sft':
        training_args_path = 'configs/training/sft_trainer.yaml'
        assert not args.question_filtering or not args.multiple_pref or not args.normalize_length, 'Question filtering, multiple preference and length normalization are not supported for SFT.'
        base_model = config.base_model_name
        model_saved_path = config.sft_model_path.format(answer_generator=args.answer_generator)
        base_tokenizer = load_tokenizer(base_model)
        # scorer
        scorer = NLIScorer(None,None,base_tokenizer,args.scoring_method,-1,max_response_tokens=128,use_tgi = True,ref_as_chosen=args.ref_as_chosen)
        
        if args.topic_generator == 'wiki':
            if args.answer_generator == 'self':
                answer_path = config.sft_answer_path.format(answer_generator = args.answer_generator)
                if not args.ref_as_chosen:
                    answer_path = answer_path.replace('.pkl','_wo_context.pkl')
                with open(answer_path,'rb') as f:
                    sft_ds = pickle.load(f)
            else:
                with open(f"data/wiki/{args.answer_generator}_answers.jsonl",'r') as f:
                    sft_ds = [json.loads(l) for l in f]

            for d in sft_ds:
                d['answer'] = d.pop('gold_answer')
            
            if args.question_filtering: # filter the dataset to build confidence around samples.
                scoring_key = 'question_hallucination'
                filtered_ds = return_question_type(sft_ds,scorer.scoring_method,type_ = 'known',score_threshold = args.question_filtering_threshold,scoring_key = scoring_key)
                if is_main_process:
                    print (f'Filtered high quality samples down to {len(filtered_ds)} samples from {len(sft_ds)} samples.')
                model_saved_path += '_qf'
                sft_ds = filtered_ds    
            loaded_ds = load_train_ds(None,base_tokenizer,sft_ds)
        else:
            loaded_ds = load_train_ds(args.topic_generator,base_tokenizer,None)
        train_ds = loaded_ds['train']
        val_ds = loaded_ds.get('val',None)
        
        ## Add in instructions with context ##
        if args.topic_generator == 'wiki' and args.answer_generator == 'self' and args.ref_as_chosen:
            answer_with_context_path = "data/wiki/context_answer.jsonl"
            with open(answer_with_context_path,'r') as f:
                answer_with_context_ds = [json.loads(l) for l in f]
            for d in answer_with_context_ds:
                d['instruction'] = format_answer(d['instruction'],document = d['document'],base=False)
                d['answer'] = d.pop('gold_answer')
            additional_train_ds = load_train_ds(None,base_tokenizer,answer_with_context_ds)['train']
            train_ds = concatenate_datasets([train_ds,additional_train_ds]) # shuffling done later when splitting train/val
        elif args.answer_generator == 'self' and not args.ref_as_chosen:
            model_saved_path += '_wo_context'

        if hasattr(config,'sft_learning_rate'):
            learning_rate = config.sft_learning_rate
        else:
            learning_rate = -1
            
    ## DPO ##
    else:
        training_args_path = 'configs/training/dpo_trainer.yaml'
        dataset_path = config.dpo_answer_path.format(answer_generator=args.answer_generator)
        if args.question_filtering: # is referring to the sft model.
            path_flag += '_qf'
        if args.multiple_pref:
            path_flag += '_multi'
        if args.normalize_length:
            path_flag += '_norm'
        base_model = config.sft_model_path.format(answer_generator='self')
        if not args.ref_as_chosen:
            base_model += '_wo_context' 
            path_flag += '_wo_context'
            dataset_path = dataset_path.replace('.pkl', '_wo_context.pkl')
        
        ## Model Path ##
        model_saved_path = config.dpo_model_path.format(answer_generator=args.answer_generator) + path_flag
        model_saved_path += f'_{args.unknown_threshold}'
        model_saved_path += f'_{args.beta}beta'
        
        base_tokenizer = load_tokenizer(base_model) # tokenizer
        
        scorer = NLIScorer(None,None,base_tokenizer,args.scoring_method,-1,max_response_tokens=128,use_tgi = True,ref_as_chosen=args.ref_as_chosen)
        
        if hasattr(config,'dpo_learning_rate'):
            learning_rate = config.dpo_learning_rate
        else:
            learning_rate = -1
        
        with open(dataset_path,'rb') as f:
            dataset = pickle.load(f)
            
        if args.question_filtering and args.ref_as_chosen:
            filtered_ds = return_question_type(dataset,args.scoring_method,'known',args.question_filtering_threshold,scoring_key = 'question_hallucination') # use a diff key
            if is_main_process:
                print (f'Filtered good samples: {len(filtered_ds)} out of {len(dataset)}')
        else:
            filtered_ds = dataset
        
        if args.unknown_threshold > 0.:
            unknown_ds = return_question_type(filtered_ds,args.scoring_method,'unknown',args.unknown_threshold) # filter to unknown samples.
        else:
            unknown_ds = dataset
        train_ds = []
        if is_main_process:
            print (f'Unknown samples: {len(unknown_ds)} out of {len(dataset)}')
        for d in unknown_ds:
            if args.multiple_pref:
                train_ds.extend(scorer.get_dpo_sample(d,multiple_pref=True,unknown_filtering=args.unknown_threshold,question_filtering=args.question_filtering_threshold))
            else:
                train_ds.append(scorer.get_dpo_sample(d,multiple_pref=False,unknown_filtering=args.unknown_threshold,question_filtering=args.question_filtering_threshold))
        train_ds = [d for d in train_ds if d is not None]
        if args.normalize_length: # normalize the preference dataset length to not have too big of a difference.
            train_ds = normalize_ds_seq_length(train_ds,base_tokenizer)
        category_counter = Counter([t['category'] for t in train_ds])
        if is_main_process:
            for k,v in category_counter.items():
                print (f'Generated {v} samples for category: {k}')
        val_ds = None
        del scorer # delete scorer to free up memory
        
    
    ## Training args ##
    train_args = yaml.safe_load(open(training_args_path, 'r')) # training args
    train_args['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if not args.save_last_only:
        train_args['save_strategy'] = 'epoch' if args.max_epochs > 0 else 'steps'
        train_args['save_total_limit'] = args.max_epochs if args.max_epochs > 0 else 5
        train_args['save_only_model'] = True
        if args.max_steps > 0:
            train_args['save_steps'] = args.max_steps//5
    if args.mode == 'dpo':
        train_args['beta'] = args.beta
        train_args['length_penalty'] = args.length_penalty
    elif args.mode == 'sft':
        if args.topic_generator == 'ultrachat':
            train_args['max_seq_length'] = 2048
        if 'tinyllama' in base_model.lower() and not args.save_last_only:
            train_args["load_best_model_at_end"] = True
            train_args["metric_for_best_model"] = "eval_loss"
            train_args["greater_is_better"] = False
            train_args['save_total_limit'] = 5 # rest need to manually look through
            if is_main_process:
                print ('Warning: Ensure Deepspeed or FSDP is not used, use the multi_gpu config instead.')
        
    os.makedirs(os.path.dirname(model_saved_path),exist_ok=True)  # make base dir

    do_train(
        ds=train_ds,
        val_ds = val_ds,
        tokenizer = base_tokenizer,
        model_name_or_path=base_model,
        batch_size=args.training_batch_size,
        max_epochs=args.max_epochs,
        max_steps = args.max_steps,
        saved_model_path=model_saved_path,
        use_peft = args.use_peft,
        peft_path = args.peft_path,
        train_args=train_args,
        learning_rate = learning_rate,
        ds_name = args.topic_generator,
    )

if __name__ == '__main__':
    main()