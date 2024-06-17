from transformers import BitsAndBytesConfig, TrainingArguments,AutoModelForCausalLM
from peft import LoraConfig,AutoPeftModelForCausalLM
from datasets import Dataset as HFDataset
from trl import DPOTrainer,SFTTrainer,DataCollatorForCompletionOnlyLM
import torch
import time
import os
from utils import get_kbit_device_map,format_response,resize_pad_embeddings
import yaml
import inspect
from accelerate import Accelerator,InitProcessGroupKwargs
import gc
from datetime import timedelta
from templates import format_answer,format_message
import random

def list_to_hf_ds(ds,model_name,tokenizer,mode ='dpo'):
    prompt_msg_fn = lambda x : format_message({'instruction':x['instruction']},tokenizer,base=False)
    if mode == 'dpo':
        prompt = [prompt_msg_fn(x) for x in ds]
        chosen = [x['chosen_ans']+tokenizer.eos_token for x in ds]
        # if not chat:
        # rejected = [x['rejected_ans']+tokenizer.eos_token for x in ds]
        # else:
        rejected = []
        for x in ds:
            if random.random() < 0.5: # we randomly dropout eos token based on https://github.com/eric-mitchell/direct-preference-optimization/issues/35
                rejected.append(x['rejected_ans']+tokenizer.eos_token)
            else:
                rejected.append(x['rejected_ans'])
        assert len(chosen) == len(rejected), 'Chosen and rejected should have same length'
        return HFDataset.from_dict({
            'prompt': prompt,
            'chosen': chosen, 
            'rejected': rejected,
        })
    elif mode == 'sft':
        formatted_items = [format_message({'instruction':format_answer(x['instruction'],x['topic'],base=False),'answer':x['answer']},tokenizer,base=False,training=True) for x in ds]
        return HFDataset.from_dict({'text':formatted_items})

    else:
        raise ValueError('Mode not supported')


def do_train(
        ds,
        val_ds = None,
        tokenizer = None,
        model_name_or_path = None,
        batch_size = 1,
        max_epochs = -1,
        max_steps = -1,
        saved_model_path=None,
        use_peft=False,
        peft_path=None,
        train_args=None,
        learning_rate=-1,
        ds_name = None,
    ):
    
    # setup quantization config
    kwargs_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    is_main_process = accelerator.is_local_main_process
    
    bnb_config = None
    if use_peft: 
        with open(peft_path, 'r') as f:
            peft_config_dict = yaml.safe_load(f)
        if peft_config_dict['quantized']:       
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    # setup Trainer args
    sig = inspect.signature(TrainingArguments.__init__)
    args_list = [param.name for param in sig.parameters.values() if param.name != 'self']
    available_trainer_args = {k:v for k,v in train_args.items() if k in args_list}
    available_trainer_args['output_dir'] = saved_model_path
    available_trainer_args['per_device_train_batch_size'] = batch_size
    available_trainer_args['per_device_eval_batch_size'] = batch_size
    num_devices = accelerator.num_processes
    if num_devices > 1 and bnb_config is not None: # if using deepspeed, cant quantize
        bnb_config = None
    total_batch_size = batch_size * num_devices
    
    if max_epochs > 0:
        available_trainer_args['num_train_epochs'] = max_epochs 
        if max_epochs < 3 and not train_args.get('load_best_model_at_end',False):
            total_steps = max_epochs * len(ds) // total_batch_size
            eval_steps = total_steps//5
            available_trainer_args['eval_steps'] = eval_steps
            available_trainer_args["evaluation_strategy"] = "steps"
        else:
            available_trainer_args["evaluation_strategy"] = "epoch"
    else:
        available_trainer_args['max_steps'] = max_steps
        available_trainer_args['warmup_steps'] = int(0.1 * max_steps)
        eval_steps = max_steps//5
        available_trainer_args['eval_steps'] = eval_steps
        available_trainer_args["evaluation_strategy"] = "steps"

    
        
    training_args = TrainingArguments(**available_trainer_args)
    if learning_rate != -1: # smaller model lower lr
        training_args.learning_rate = learning_rate 
    
    # Setup train/eval ds #
    if isinstance(ds,list):
        full_ds = list_to_hf_ds(ds,model_name_or_path,tokenizer,mode=train_args['mode'])
    else:
        full_ds = ds
    if train_args['do_eval']:
        if val_ds is None:
            eval_size = int(train_args['eval_split'] * len(full_ds))
            full_ds = full_ds.shuffle(seed=42)
            train_ds = full_ds.select(range(len(full_ds)-eval_size))
            val_ds = full_ds.select(range(len(full_ds)-eval_size,len(full_ds)))
        else:
            train_ds = full_ds
            if isinstance(ds,list):
                val_ds = list_to_hf_ds(val_ds,model_name_or_path,tokenizer,mode=train_args['mode'])
    else:
        val_ds = full_ds.select(range(100))
        train_ds = full_ds
    
    ## Print out first sample
    
    random_train_sample = train_ds.shuffle(seed=42).select(range(5))
    if is_main_process:
        for rs in random_train_sample:
            if train_args['mode'] == 'sft':
                print ('example: ',rs['text'])
            elif train_args['mode'] == 'dpo':
                print ('prompt: ',rs['prompt'])
                print ('chosen: ',rs['chosen'])
                print ('rejected: ',rs['rejected'])
    ## Model args
    model_kwargs = dict(
        revision='main',
        device_map = None if not bnb_config else get_kbit_device_map(),
        attn_implementation = 'flash_attention_2',
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache = False if training_args.gradient_checkpointing else True,
        torch_dtype = train_args['torch_dtype'] if train_args.get('torch_dtype',None) in ["auto", None] else getattr(torch, train_args['torch_dtype'])
    )

    if train_args['mode'] == 'sft':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,**model_kwargs)
        if tokenizer.pad_token_id is None or tokenizer.pad_token == tokenizer.eos_token:
            resize_pad_embeddings(model,tokenizer)
    else:
        if 'chat' in model_name_or_path.lower() or 'zephyr' in model_name_or_path.lower():
            tokenizer.pad_token = tokenizer.eos_token # in dpo, setting this is not a problem.
    # if train_args['mode'] == 'dpo':
        model = model_name_or_path
        model_ref=model
        ref_model_kwargs = model_kwargs
    
    # Setup PEFT config
    if use_peft:
        peft_config = LoraConfig(
            r=peft_config_dict['lora_r'],
            lora_alpha=peft_config_dict['lora_alpha'],
            lora_dropout=peft_config_dict['lora_dropout'],
            target_modules=peft_config_dict['lora_layers'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_ref = None
        ref_model_kwargs=None
        training_args.learning_rate = training_args.learning_rate* peft_config_dict['learning_rate_multiplier'] # peft uses higher learning rate, default * 10
        training_args.optim = peft_config_dict['optim']
        
    else:
        peft_config = None
    
    if training_args.gradient_checkpointing and use_peft:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False} # use this when gradient checkpointing is used.
    
    if train_args['mode'] == 'dpo':
        trainer = DPOTrainer(
            model,
            model_ref,
            model_init_kwargs = model_kwargs,
            ref_model_init_kwargs = ref_model_kwargs,
            args=training_args,
            beta=train_args['beta'],
            train_dataset=train_ds,
            eval_dataset = val_ds,
            tokenizer=tokenizer,
            max_length=train_args['max_length'],
            max_prompt_length=train_args['max_prompt_length'],
            peft_config=peft_config,
            alpha = train_args['length_penalty'],
            add_eos_token = False,
        )
        
        ## Fix for FSDP in https://github.com/huggingface/trl/issues/1147
        if trainer.is_fsdp_enabled:
            prepared_model = trainer._wrap_model(
            trainer.model, training=True, dataloader=None)
            
            if hasattr(trainer.lr_scheduler, "step"):
                prepared_model, trainer.optimizer = trainer.accelerator.prepare(
                    prepared_model, trainer.optimizer
                )
            else:
                (
                    prepared_model,
                    trainer.optimizer,
                    trainer.lr_scheduler,
                ) = trainer.accelerator.prepare(
                    prepared_model, trainer.optimizer, trainer.lr_scheduler
                )
            trainer.model_wrapped = prepared_model
            if trainer.is_fsdp_enabled:
                trainer.model = prepared_model
            if trainer.ref_model is not None:
                trainer.ref_model = trainer.accelerator.prepare_model(trainer.ref_model)

            trainer.accelerator.prepare_model = lambda model, *args, **kwargs: model 

        
    else:
        dataset_kwargs = {'add_special_tokens':True}
        if ds_name == 'ultrachat':
            dataset_kwargs['append_concat_token'] = False # dont append concat token since chat format already have eos token
            packing = True
            data_collator = None
        else:    
            # if 'chat' in model_name_or_path.lower() or 'zephyr' in model_name_or_path.lower():
            response_template = "<|assistant|>\n"
            # else:
            # # _,_,response_template = format_instruction_response(prompt_format)
            #     response_template = "### Response:"
            response_template_ids = tokenizer.encode('\n'+response_template, add_special_tokens=False)[2:]
            data_collator = DataCollatorForCompletionOnlyLM(response_template =response_template_ids, tokenizer=tokenizer)
            packing = False

        trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=train_args['max_seq_length'],
        tokenizer=tokenizer,
        packing=packing,
        peft_config=peft_config,
        dataset_text_field = 'text',
        dataset_num_proc=64,
        dataset_kwargs = dataset_kwargs,
        data_collator = data_collator,
        )
    trainer.train()
    if saved_model_path is None:
        saved_model_path = "trained_" + str(time.time()).replace('.', '_')
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(saved_model_path)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    
    # clear mem
    del trainer.model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    del train_ds
    if val_ds is not None:
        del val_ds
    
    if use_peft: # merge it if peft is used
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
                    saved_model_path,
                    torch_dtype=torch.float16,
                    device_map = None)  
        merged_model = peft_model.merge_and_unload(progressbar=True)
        for os_file in os.listdir(saved_model_path): # remove the adapter files.
            if Accelerator().is_main_process:
                if 'adapter' in os_file:
                    os.remove(os.path.join(saved_model_path, os_file))
        merged_model.save_pretrained(saved_model_path)
    
        