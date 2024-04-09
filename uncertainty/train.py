import os
import json
from accelerate import Accelerator
import argparse
import yaml
from types import SimpleNamespace
from self_learning_training import do_train
from datasets import load_dataset
from utils import *
from data_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="",required=True)
    parser.add_argument("--answer_generator", type=str, default="self",required=True)
    parser.add_argument("--scoring_method", type=str, default="BSDetector",required=True)
    parser.add_argument("--topic_generator", type=str, default="fixed_truthful_qa",required=True)
    parser.add_argument("--max_epochs",  type = int,default = -1)
    parser.add_argument("--max_steps",  type = int,default = -1)
    parser.add_argument("--training_batch_size",  type = int,default = 4)
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--peft_path",  type = str,default = 'configs/training/lora.yaml',help = 'get lora parameters')
    parser.add_argument("--training_args_path",  type = str,default = 'configs/training/dpo_trainer.yaml',help = 'get train params, default DPO')
    parser.add_argument("--use_peft",  type = bool,default = False,help = 'use peft')
    args = parser.parse_args()
    
    assert (args.max_epochs > 0 and args.max_steps < 0) or (args.max_epochs < 0 and args.max_steps > 0), 'Either max_epochs or max_steps should be set.'
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    
    # if 'fixed' in args.topic_generator or 'predefined' in args.topic_generator or 'oracle' in args.topic_generator: # data config.
    #     ds_name = '_'.join(args.topic_generator.split('_')[1:]) 
    # elif 'wiki' in args.topic_generator:
    #     ds_name = 'wiki'
    # else:
    #     ds_name = args.topic_generator
        
    # ds_config_path = f'configs/data/{ds_name}.yaml'
    # with open(ds_config_path,'r') as f:
    #     ds_config = yaml.safe_load(f)
    
    if args.scoring_method == 'BSDetector':
        scoring_name = 'conf' 
    elif args.scoring_method == 'semantic_consistency':
        scoring_name = 'entropy'
    else:
        scoring_name = 'hallu'
    
    base_tokenizer = load_tokenizer(config.model_name,truncation_side='left')
    if hasattr(config,"train_dataset_path"): # else is SFT.
        config.train_dataset_path = config.train_dataset_path.format(topic_generator=args.topic_generator,
                                                                 answer_generator=args.answer_generator,
                                                                scoring_name=scoring_name)
    
        
        assert os.path.exists(config.train_dataset_path), f'{config.train_dataset_path} not found'
        with open(config.train_dataset_path,'r') as f:
            train_ds = [json.loads(line) for line in f]
        
        ## Filter training ds here.
        if args.filter_size < 1.0:
            filtered_len = int(len(train_ds)*args.filter_size)
            train_ds = sorted(train_ds,key = lambda x:x['question_score'],reverse=True)[:filtered_len] # take highest uncertainty.
        val_ds = None
    else:
        loaded_ds = load_train_ds(args.topic_generator,base_tokenizer)
        train_ds = loaded_ds['train']
        val_ds = loaded_ds.get('val',None)
    
    config.model_path = config.model_path.format(topic_generator=args.topic_generator,
                                                    answer_generator=args.answer_generator,
                                                    scoring_name=scoring_name,
                                                    filter_size = int(100*args.filter_size))
    
    if not args.use_peft:
        config.model_path = config.model_path.replace('model_checkpoints','model_checkpoints_full')
    os.makedirs(os.path.dirname(config.model_path),exist_ok=True)
    

    ## DEL HERE
    # train_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",split='train_prefs')
    # train_ds = train_ds.shuffle(seed=42).select(range(30000))
    
    # def map_fn(example,tokenizer):
    #     prompt_msg = example['chosen'][:-1]
    #     # if prompt_msg[0]['role'] != 'system':
    #     #     prompt_msg.insert(0,{'role':'system','content':''})
    #     chosen_messages = example["chosen"][-1]['content']
    #     rejected_messages = example["rejected"][-1]['content']
    #     example["chosen"] = chosen_messages + "</s>"
    #     example["rejected"] = rejected_messages + "</s>"
    #     example["prompt"] = tokenizer.apply_chat_template(prompt_msg, tokenize=False,add_generation_prompt = True)
    #     return example
    # train_ds = train_ds.map(map_fn,fn_kwargs={'tokenizer':base_tokenizer},num_proc=64,desc="Formatting comparisons with prompt template")
    # config.model_path += "_test"
    ## DEL END
    
    print ('**Training model**')
    do_train(
        ds=train_ds,
        val_ds = val_ds,
        tokenizer = base_tokenizer,
        model_name_or_path=config.model_name,
        batch_size=args.training_batch_size,
        max_epochs=args.max_epochs,
        max_steps = args.max_steps,
        saved_model_path=config.model_path,
        use_peft = args.use_peft,
        peft_path = args.peft_path,
        train_args_path=args.training_args_path,
        learning_rate = config.learning_rate if hasattr(config,'learning_rate') else -1, # tinyllama use bigger lr.
    )

if __name__ == '__main__':
    main()