from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset as HFDataset
import bitsandbytes as bnb
from trl import DPOTrainer
from tqdm import tqdm
import pandas as pd
import statistics
import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from collections import defaultdict
from uncertainty.utils import tgi_to_gen_kwargs,HF_generate,tgi_generate
from uncertainty.templates import format_question_generation
from uncertainty.scorer import LLMJudge
import yaml


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

def list_to_hf_ds(ds,prompt_fn,test=False):
    if not test:
        return HFDataset.from_dict({
            'prompt': [prompt_fn(x['instruction']) for x in ds],
            'chosen': [x['chosen_ans'] for x in ds],
            'rejected': [x['rejected_ans'] for x in ds]
        })
    else:
        return HFDataset.from_dict({
            'prompt': [prompt_fn(x['instruction']) for x in ds],
            'pre_response':[x['pre_response'] for x in ds],
            'chosen_ans': [x['chosen_ans'] for x in ds],
        })

def do_train(
        ds,
        questions_with_hallucination,
        prompt_format_fn,
        extract_response_fn,
        tokenizer,
        model_name_or_path,
        batch_size = 1,
        max_epochs = 80,
        lr = 3e-5,
        deterministic = True,
        saved_model_path=None,
        max_response_len=256,
        use_peft=False,
        peft_path=None,
    ):
    # train_ds = create_train_ds(ds, questions_with_hallucination, prompt_format_fn)
    train_ds = list_to_hf_ds(ds,prompt_format_fn)
    if use_peft:
        with open(peft_path, 'r') as f:
            peft_config = yaml.safe_load(f)
        if peft_config['quantized']:       
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
    else:
        bnb_config=None
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype='auto',
    )
    base_model.enable_input_require_grads()
    base_model.config.use_cache = False
    
    if use_peft:
        ## get linear layers for lora
        linear_modules = set()
        for n,m in base_model.named_modules():
            if isinstance(m,torch.nn.Linear):
                linear_modules.add(n.split('.')[-1])
        
        peft_config = LoraConfig(
            r=peft_config['lora_r'],
            lora_alpha=peft_config['lora_alpha'],
            lora_dropout=peft_config['lora_dropout'],
            target_modules=list(linear_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        print (f'LORA: Training these layers: {linear_modules}')
    else:
        peft_config = None
    
    
    model_ref=None

    training_args = TrainingArguments(
        output_dir=saved_model_path,
        overwrite_output_dir=True,
        full_determinism=deterministic,
        do_train=True,
        do_eval=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        per_device_train_batch_size=batch_size,
        optim='adamw_bnb_8bit',
        learning_rate=lr,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bf16 = True,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.1,
        num_train_epochs=max_epochs,
        save_total_limit=1,
        report_to='tensorboard',
        disable_tqdm=False,
        push_to_hub=False,
    )
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False} # use this when gradient checkpointing is used.
    
    dpo_trainer = DPOTrainer(
        base_model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=max_response_len,
        peft_config=peft_config,
    )
    dpo_trainer.train()
    if saved_model_path is None:
        saved_model_path = "trained_" + str(time.time()).replace('.', '_')
    # dpo_trainer.save_model(saved_model_path)
    dpo_trainer.model.save_pretrained(saved_model_path)
    dpo_trainer.tokenizer.save_pretrained(saved_model_path)
    if training_args.push_to_hub:
        dpo_trainer.push_to_hub()
    
    del dpo_trainer
    if use_peft: # merge it if peft is used
        if bnb_config: # reload in 16bit
            del base_model
            base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=None,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            )
        peft_model = PeftModel.from_pretrained(
                model=base_model,
                model_id = saved_model_path,
                is_trainable=False,
            )
        base_model = peft_model.merge_and_unload()
        base_model = base_model.cpu()
        for os_file in os.listdir(saved_model_path): # remove the adapter files.
            if 'adapter' in os_file:
                os.remove(os.path.join(saved_model_path, os_file))
        base_model.save_pretrained(saved_model_path)
    # wandb.finish()


def do_test(ds,
            prompt_fn_dict,
            model,
            tokenizer,
            scorer,
            batch_size,
            gen_kwargs,
            out_file,
            test_keys,
            **kwargs):
    
    ques_prompt_fn = prompt_fn_dict['prompt_fn']['question_gen']
    ans_prompt_fn = prompt_fn_dict['prompt_fn']['answer_gen']
    extract_ans_fn = prompt_fn_dict['extract_ans_fn']
    
    test_ds = list_to_hf_ds(ds,ans_prompt_fn,test=True)
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
            
    
    ## Test answer performance ##
    cost = 0.0
    if 'answer_performance' in test_keys:
        llmjudge = LLMJudge() # default uses gpt4
        win_rate,cost = llmjudge.evaluate(compare_ans)
        results_dict['answer_win_rate'] = round(win_rate,3)
        
    ## Test confidence of questions generated ##
    if 'question_confidence' in test_keys:
        questions_kwargs = gen_kwargs['questions']
        if not use_tgi:
            questions_kwargs = tgi_to_gen_kwargs(questions_kwargs)
    
        topics = kwargs['topics']
        for idx in tqdm(range(0,len(topics),batch_size),total=len(topics)//batch_size,desc='Generating question confidence'):
            topic_batch = topics[idx:idx+batch_size]
            formatted_topic_inp = [ques_prompt_fn(format_question_generation(t)) for t in topic_batch]
            gen_questions = HF_generate(formatted_topic_inp,model,tokenizer,questions_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            if not use_tgi:
                gen_questions = [x['text'] for x in gen_questions]
            
            if scorer.scoring_method == 'BSDetector':
                question_ref_ans = HF_generate([ans_prompt_fn(x) for x in gen_questions],model,tokenizer,ref_answer_gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
            else:
                question_ref_ans = [None for _ in gen_questions]
            question_sampled_ans = HF_generate([ans_prompt_fn(x) for x in gen_questions],model,tokenizer,sampled_answer_gen_kwargs,extract_ans_fn,max_length=4096,use_tgi=use_tgi)
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
        
    if out_file is not None:
        with open(out_file,'w') as f:
            f.write(f'Metric type: {scoring_key}\n')
            for k,v in results_dict.items():
                if isinstance(v,float) or isinstance(v,int):
                    f.write(f"{k}:{np.round(v,3)}\n")
                else:
                    f.write(f"{k}:{v}\n")
            f.write(f'LLMJudge eval cost: {cost}')
    return results_dict
        
    
        