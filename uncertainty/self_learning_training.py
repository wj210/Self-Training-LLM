from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel,AutoPeftModelForCausalLM
from datasets import Dataset as HFDataset
from trl import DPOTrainer,SFTTrainer,DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import pandas as pd
import statistics
import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from collections import defaultdict
from uncertainty.utils import tgi_to_gen_kwargs,HF_generate,get_kbit_device_map,format_response,format_instruction_response
from uncertainty.templates import format_question_generation
from uncertainty.scorer import LLMJudge
import yaml
from sklearn.metrics import roc_auc_score
import inspect
from accelerate import Accelerator


def create_train_ds(ds, questions_with_hallucination, prompt_format_fn):
    prompts = [x['prompt'] for x in questions_with_hallucination]
    passages = [x['passage'] for x in questions_with_hallucination]
    preds_df = pd.DataFrame({
        'prompt': prompts,
        'passage': passages
    }).sort_values(['prompt'], ascending=False).fillna('-')
    preds_df['prompt'] = preds_df['prompt'].apply(str).str.strip()
    preds_df['passage'] = preds_df['passage'].apply(str).str.strip()
    docs_df = pd.DataFrame(ds)[['prompt', 'answer_score', 'answer_text', 'summary_text']]
    docs_df['prompt'] = docs_df['prompt'].apply(str).str.strip()
    docs_df['answer_text'] = docs_df['answer_text'].apply(str).str.strip()
    docs_df['summary_text'] = docs_df['summary_text'].apply(str).str.strip()
    docs_df['answer_score'] = docs_df['answer_score'].astype(float)
    all_df = pd.merge(docs_df, preds_df, on='prompt', how='left')
    all_df = all_df.sort_values(['prompt', 'answer_score'], ascending=False)
    all_df = all_df.groupby('prompt').head(1) # take answer with top score

    print(all_df.head(20))

    prompt_list = [prompt_format_fn(x) for x in all_df['prompt'].values.tolist()]
    tmp_answer = all_df['answer_text'].values.tolist()
    tmp_summary = all_df['summary_text'].values.tolist()
    chosen_list = [a.strip() + '. ' + s.strip() for a, s in zip(tmp_answer, tmp_summary)]
    rejected_list = all_df['passage'].values.tolist()
    return HFDataset.from_dict({
        'prompt': prompt_list,
        'chosen': chosen_list,
        'rejected': rejected_list
    })

def list_to_hf_ds(ds,model_name,tokenizer,test=False,mode ='dpo'):
    prompt_msg_fn = lambda x : format_response([{'role':'user','content':x}],model_name,tokenizer,mode='answer')

    if not test:
        if mode == 'dpo':
            return HFDataset.from_dict({
                'prompt': [prompt_msg_fn(x['instruction']) for x in ds],
                'chosen': [x['chosen_ans'] for x in ds],
                'rejected': [x['rejected_ans'] for x in ds]
            })
        elif mode == 'sft':
            # message = [format_response([{'role':'user','content':x['instruction']},
            #             {'role':'assistant','content':x['chosen_ans']}],model_name,tokenizer,mode='label') for x in ds] # dont add generation prompt
            # return HFDataset.from_dict({'prompt':message})
            return HFDataset.from_dict({'prompt':[x['instruction'] for x in ds],
                                        'completion':[x['chosen_ans'] for x in ds]
                                        })
            
        else:
            raise ValueError('Mode not supported')
    else:
        return HFDataset.from_dict({
            'prompt': [prompt_msg_fn(x['instruction']) for x in ds],
            'pre_response':[x['pre_response'] for x in ds],
            'chosen_ans': [x['chosen_ans'] for x in ds],
        })

def do_train(
        ds,
        tokenizer,
        model_name_or_path,
        batch_size = 1,
        max_epochs = -1,
        max_steps = -1,
        saved_model_path=None,
        max_response_len=256,
        use_peft=False,
        peft_path=None,
        train_args_path=None,
        learning_rate=-1,
    ):
    
    # setup quantization config
    bnb_config = None
    if use_peft: 
        with open(peft_path, 'r') as f:
            peft_config = yaml.safe_load(f)
        if peft_config['quantized']:       
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    # setup Trainer args
    train_args = yaml.safe_load(open(train_args_path, 'r'))
    sig = inspect.signature(TrainingArguments.__init__)
    args_list = [param.name for param in sig.parameters.values() if param.name != 'self']
    available_trainer_args = {k:v for k,v in train_args.items() if k in args_list}
    available_trainer_args['output_dir'] = saved_model_path
    available_trainer_args['per_device_train_batch_size'] = batch_size
    available_trainer_args['per_device_eval_batch_size'] = batch_size
    if max_epochs > 0:
        available_trainer_args['num_train_epochs'] = max_epochs 
        available_trainer_args['warmup_ratio'] = 0.1
        available_trainer_args['evaluation_strategy'] = 'epoch'
    else:
        available_trainer_args['max_steps'] = max_steps
        available_trainer_args['warmup_steps'] = int(0.1 * max_steps)
        available_trainer_args['evaluation_strategy'] = 'steps'
        available_trainer_args['eval_steps'] = int(0.2 * max_steps)
        
    training_args = TrainingArguments(**available_trainer_args)
    if train_args['mode'] == 'sft': # smaller model lower lr
        training_args.learning_rate = learning_rate if learning_rate != -1 else training_args.learning_rate
    
    # Setup train/eval ds #
    full_ds = list_to_hf_ds(ds,model_name_or_path,tokenizer,mode=train_args['mode']).shuffle(seed=42)
    eval_size = int(train_args['eval_split'] * len(full_ds))
    train_ds = full_ds.select(range(len(full_ds)-eval_size))
    val_ds = full_ds.select(range(len(full_ds)-eval_size,len(full_ds)))
    
    ## Print out first sample
    random_train_sample = train_ds[0]
    random_eval_sample = val_ds[0]
    if train_args['mode'] == 'sft':
        print ('train prompt: ',random_train_sample['prompt'])
        print ('train answer: ',random_train_sample['completion'])
        print ('eval prompt: ',random_eval_sample['prompt'])
        print ('eval answer: ',random_eval_sample['completion'])
    elif train_args['mode'] == 'dpo':
        print ('train prompt: ',random_train_sample['prompt'])
        print ('train chosen: ',random_train_sample['chosen'])
        print ('train rejected: ',random_train_sample['rejected'])
        print ('eval prompt: ',random_eval_sample['prompt'])
        print ('eval chosen: ',random_eval_sample['chosen'])
        print ('eval rejected: ',random_eval_sample['rejected'])
        
    ## Model args
    model_kwargs = dict(
        torch_dtype='auto',
        device_map = None if not bnb_config else get_kbit_device_map(),
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache = False if training_args.gradient_checkpointing else True,
    )
    ref_model_kwargs = model_kwargs

    model = model_name_or_path # initalize as path
    model_ref=model
    
    # Setup PEFT config
    if use_peft:
        peft_config = LoraConfig(
            r=peft_config['lora_r'],
            lora_alpha=peft_config['lora_alpha'],
            lora_dropout=peft_config['lora_dropout'],
            target_modules=peft_config['lora_layers'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_ref = None
        ref_model_kwargs = None
        
    else:
        peft_config = None
    
    if training_args.gradient_checkpointing and use_peft:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False} # use this when gradient checkpointing is used.
    
    if train_args['mode'] == 'dpo':
        trainer = DPOTrainer(
            model,
            model_ref,
            model_init_kwargs = model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            beta=train_args['beta'],
            train_dataset=train_ds,
            eval_dataset = val_ds,
            tokenizer=tokenizer,
            max_length=train_args['max_prompt_length'] + max_response_len,
            max_prompt_length=train_args['max_prompt_length'],
            peft_config=peft_config,
        )
    else:
        format_template,response_template = format_instruction_response(model)
        # response_template = tokenizer.encode('\n'+response_template,add_special_tokens=False)[2:] # add '\n to remove leading char.
        def format_prompt_fn(example):
            output_texts = []
            for i in range(len(example['prompt'])):
                text = format_template.format(instruction=example['prompt'][i],output=example['completion'][i])
                output_texts.append(text)
            return output_texts
        
        data_collator = DataCollatorForCompletionOnlyLM(response_template,tokenizer=tokenizer) # only trained on output
        
        trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=train_args['max_prompt_length'] + max_response_len,
        tokenizer=tokenizer,
        packing=False,
        peft_config=peft_config,
        # dataset_text_field = 'prompt'
        data_collator = data_collator,
        formatting_func=format_prompt_fn,
        )
    trainer.train()
    if saved_model_path is None:
        saved_model_path = "trained_" + str(time.time()).replace('.', '_')
    trainer.save_model(saved_model_path)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    
    del trainer
    if use_peft: # merge it if peft is used
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
                    saved_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True)  
        base_model = peft_model.merge_and_unload()
        base_model = base_model.cpu()
        for os_file in os.listdir(saved_model_path): # remove the adapter files.
            # if Accelerator().is_main_process:
            if 'adapter' in os_file:
                os.remove(os.path.join(saved_model_path, os_file))
            base_model.save_pretrained(saved_model_path)
    # wandb.finish()


def do_test(ds,
            model,
            model_name,
            tokenizer,
            scorer,
            batch_size,
            gen_kwargs,
            out_file,
            test_keys,
            few_shots,
            extract_ans_fn,
            **kwargs):
    
    
    test_ds = list_to_hf_ds(ds,model_name,tokenizer,test=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    ## Convert TGI kwargs to standard HF generate kwargs
    ref_answer_gen_kwargs = gen_kwargs['ref_answer']
    sampled_answer_gen_kwargs = gen_kwargs['sample_answers']
    use_tgi = kwargs['use_tgi']
    if not use_tgi:
        sampled_answer_gen_kwargs = tgi_to_gen_kwargs(sampled_answer_gen_kwargs)
        ref_answer_gen_kwargs = tgi_to_gen_kwargs(ref_answer_gen_kwargs)
        num_sampled_seq = sampled_answer_gen_kwargs['num_return_sequences']
        model.eval()
    else:
        num_sampled_seq = sampled_answer_gen_kwargs['best_of']
    results_dict = defaultdict(list)
    
    ## Set up scoring key for different scoring method, entropy or confidence
    score_key = {'BSDetector':'confidence',
                'semantic_consistency':'overall_entropy'}
    scoring_key = score_key[scorer.scoring_method]
    
    ## Test confidence of answers generated ##
    compare_ans = []
    instr_2_confidence = {}
    if 'answer_confidence' in test_keys or 'answer_performance' in test_keys:
        for batch in tqdm(test_dl,total=len(test_dl),desc='Generating answer confidence'):
            instructions = [x for x in batch['prompt']]
            post_response = []
            if scorer.scoring_method == 'BSDetector' or 'answer_performance' in test_keys:
                ref_answers = HF_generate(instructions,model,tokenizer,ref_answer_gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            else:
                ref_answers = [None for _ in instructions]
            sampled_answers = HF_generate(instructions,model,tokenizer,sampled_answer_gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            if not use_tgi:
                batched_answers = [(instructions[i],ref_answers[i],sampled_answers[i*num_sampled_seq:(i+1)*num_sampled_seq]) for i in range(len(ref_answers))]
                if ref_answers[0] is not None:
                    post_response = [x['text'] for x in ref_answers]  
                    
            else:
                batched_answers = [(instructions[i],ref_answers[i],sampled_answers[i]) for i in range(len(ref_answers))]
                if ref_answers[0] is not None:
                    post_response = [x.generated_text for x in ref_answers]
            
            if len(post_response) > 0:
                compare_ans.extend([{'instruction':instr,'pre_response':pre,'post_response':post} for instr,pre,post in zip(batch['prompt'],batch['pre_response'],post_response)])
                
            confidence_score_dict = [scorer.get_score(instr,ra,sa) for instr,ra,sa in batched_answers]
            results_dict['post_answer_score'].extend([cd[scoring_key] for cd in confidence_score_dict])
            for cd in confidence_score_dict:
                instr_2_confidence[cd['instruction']] = cd[scoring_key]
    
    ## Test answer performance ##
    cost = 0.0
    if 'answer_performance' in test_keys:
        llmjudge = LLMJudge() # default uses gpt4
        win_rate,win_rate_dict,cost = llmjudge.evaluate(compare_ans)
        results_dict['answer_win_rate'] = round(win_rate,3)
    
    ## Evaluate the AUROC of the confidence score, ie how well calibrated or aligned the confidence/uncertainty is with perf ##
        if 'answer_confidence' in test_keys:
            ans_acc,ans_conf = [],[]
            for instr,conf in instr_2_confidence.items():
                ans_acc.append(win_rate_dict[instr])
                ans_conf.append(conf)
            try:
                results_dict['auroc'] = round(roc_auc_score(ans_acc,ans_conf),3)
            except Exception as e:
                print (e)
                results_dict['auroc'] = 0.0
    
    ## Test confidence of questions generated ##
    if 'question_confidence' in test_keys:
        questions_kwargs = gen_kwargs['questions']
        if not use_tgi:
            questions_kwargs = tgi_to_gen_kwargs(questions_kwargs)
    
        topics = kwargs['topics']
        for idx in tqdm(range(0,len(topics),batch_size),total=len(topics)//batch_size,desc='Generating question confidence'):
            topic_batch = topics[idx:idx+batch_size]
            formatted_topic_inp = [format_response([{'role':'user','content':format_question_generation(t)}],model_name,tokenizer,mode='question') for t in topic_batch]
            gen_questions = HF_generate(formatted_topic_inp,model,tokenizer,questions_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            if not use_tgi:
                gen_questions = [x['text'] for x in gen_questions]
            
            if scorer.scoring_method == 'BSDetector':
                question_ref_ans = HF_generate([format_response([{'role':'user','content':x}],model_name,tokenizer,mode='answer') for x in gen_questions],model,tokenizer,ref_answer_gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            else:
                question_ref_ans = [None for _ in gen_questions]
            question_sampled_ans = HF_generate([format_response([{'role':'user','content':x}],model_name,tokenizer,mode='answer') for x in gen_questions],model,tokenizer,sampled_answer_gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            if not use_tgi:
                batched_qn_answers = [(gen_questions[i],question_ref_ans[i],question_sampled_ans[i*num_sampled_seq:(i+1)*num_sampled_seq]) for i in range(len(question_ref_ans))]     
            else:
                batched_qn_answers = [(gen_questions[i],question_ref_ans[i],question_sampled_ans[i]) for i in range(len(question_ref_ans))] 

            question_post_score_dict = [scorer.get_score(gen_instr,qra,qrsa) for gen_instr,qra,qrsa in batched_qn_answers]
            question_post_score_dict = [x[scoring_key] for x in question_post_score_dict if x is not None]
            results_dict['post_question_score'].extend(question_post_score_dict)
    
    removed_question_len = len(topics) - len(results_dict['post_question_score'])
    print (f'Removed {removed_question_len} questions')
    
    results_dict['post_question_score'] = statistics.mean(results_dict['post_question_score'])
    
    if 'question_confidence' in test_keys:
        results_dict['question_score_diff'] = (results_dict['post_question_score'] - kwargs['pre_question_score'])
        results_dict['pre_question_score'] = kwargs['pre_question_score']
    
    if 'answer_confidence' in test_keys:
        answer_score_diff = np.mean([post - pre for post,pre in zip(results_dict[f'post_answer_score'],kwargs['pre_answer_score'])])
        results_dict['answer_score_diff'] = answer_score_diff
        results_dict['post_answer_score'] = statistics.mean(results_dict['post_answer_score'])
        results_dict['pre_answer_score'] = statistics.mean(kwargs['pre_answer_score'])
        
    with open(out_file + '.txt','w') as f:
        f.write(f'Metric type: {scoring_key}\n')
        for k,v in results_dict.items():
            if isinstance(v,float) or isinstance(v,int):
                f.write(f"{k}:{np.round(v,3)}\n")
            else:
                f.write(f"{k}:{v}\n")
        f.write(f'LLMJudge eval cost: {cost}')
    return results_dict
        
    
        