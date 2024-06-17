import os
import json
import yaml
from scorer import LLMJudge,NLIScorer
from huggingface_hub import InferenceClient
from utils import *
from data_utils import *
import argparse
from types import SimpleNamespace
from copy import deepcopy
from tqdm import tqdm
from templates import generation_examples,format_message,format_answer

def get_few_shot_instrs(category,max_few_shots=5):
    return [{'instruction':format_answer(f['instruction'],base=True),'answer':f['answer']} for f in generation_examples[:max_few_shots]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082,help = 'port for TGI')
    parser.add_argument("--config_path", type=str, default="configs/model/tinyllama.yaml")
    parser.add_argument("--mode", type=str, default="sft")
    parser.add_argument("--question_filtering", action='store_true')
    parser.add_argument("--multiple_pref",  action = 'store_true',help= 'Create multiple preference samples per instruction')
    parser.add_argument("--unknown_threshold",  default = 0.5,help= 'score to select the split unknown samples')
    parser.add_argument("--answer_generator", type=str, default="self")
    parser.add_argument("--beta",  default = 0.3,help= 'beta value for DPO')
    parser.add_argument("--num_samples", type=int, default=10,help = 'port for TGI')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--use_tgi",  action = 'store_true',help= 'use tgi')
    parser.add_argument("--known",  action = 'store_true',help= 'test on known samples (training set)')
    args = parser.parse_args()
    
    seed_all(42)
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    
    ds_config_path = 'configs/data/wiki.yaml'
    
    with open(ds_config_path,'r') as f:
        ds_config = yaml.safe_load(f)
    
    gen_kwargs = ds_config['gen_kwargs']
    gen_kwargs['repetition_penalty'] = 1.1
    sample_kwargs = deepcopy(gen_kwargs)
    if 'temperature' in gen_kwargs:
        gen_kwargs.pop('temperature')
    sample_kwargs['do_sample'] = True
    sample_kwargs['best_of'] = args.num_samples
    sample_kwargs['details'] = True

    if not args.use_tgi:
        sample_kwargs = tgi_to_gen_kwargs(sample_kwargs)
    
    path_flag = ''
    if args.question_filtering:
        path_flag += '_qf'
    if args.multiple_pref:
        path_flag += '_multi'
    
    
    base = False
    use_flash=True
    if args.mode == 'dpo':
        path_flag += f'_{args.unknown_threshold}'
        path_flag += f'_{args.beta}beta'
        model_path = config.dpo_model_path.format(answer_generator=args.answer_generator) + path_flag
    elif args.mode in['sft','dola','rag']:
        model_path = config.sft_model_path.format(answer_generator=args.answer_generator) + path_flag
        if args.mode == 'dola':
            assert not args.use_tgi, 'DOLA only for base model and cannot use tgi.'
            gen_kwargs['repetition_penalty'] = 1.2
            gen_kwargs['dola_layers'] = 'high'
            use_flash = False
    
    elif args.mode == 'base':
        model_path = config.base_model_name
        base=True
    
    config.result_path = config.result_path.format(mode=args.mode,answer_generator=args.answer_generator).replace('.txt',f'{path_flag}.txt')
    response_path = config.result_path.replace('test_results','responses').replace('.txt','.jsonl')
    os.makedirs(os.path.dirname(response_path),exist_ok=True)
    
    if args.use_tgi:
        client = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    else:
        client = load_hf_model(model_path,use_flash=use_flash).eval()
    tokenizer = load_tokenizer(model_path,padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_name = args.config_path.split('/')[-1].split('.yaml')[0].strip()
    
    if not args.known:
        test_ds = [json.loads(l) for l in open('data/wiki/test.jsonl','r')]
    else:
        knwn_ds_path = f"data/wiki/{model_name}_known.jsonl"
        if not os.path.exists(knwn_ds_path):
            test_ds = pickle.load(open(config.dpo_answer_path.format(answer_generator="self"),'rb'))
            test_ds = return_question_type(test_ds,'SelfCheckGPT','known',0.5) # return known
            random.shuffle(test_ds)
            test_ds = test_ds[:200] 
            test_ds = [{'instruction':x['instruction'],'category':x['category'],'document':x['document']} for x in test_ds]
            with open(knwn_ds_path,'w') as f: # save to ensure same split across different models.
                for ds in test_ds:
                    f.write(json.dumps(ds)+'\n')
        else:
            test_ds = [json.loads(l) for l in open(knwn_ds_path,'r')]
        response_path = response_path.replace('.jsonl','_known.jsonl')

    
    response_list = []
    for batch_idx in tqdm(range(0,len(test_ds),args.batch_size),total = len(test_ds)//args.batch_size,desc = f"Generate responses for {model_name}_{args.mode}{path_flag}"):
        test_batch = test_ds[batch_idx:batch_idx+args.batch_size]
        instrs = [x['instruction'] for x in test_batch]
        if base: # include few_shot
            category = [x['category'] for x in test_batch]
            cat_fs = [get_few_shot_instrs(cat) for cat in category]
        else:
            cat_fs = []
        if args.mode == 'rag':
            documents = [x['document'] for x in test_batch]
            formatted_instrs = [format_answer(instr,base=base,document = docu) for instr,docu in zip(instrs,documents)]
        else:
            formatted_instrs = [format_answer(instr,base=base) for instr in instrs]
        
        if len(cat_fs) == 0:
            formatted_instrs = [format_message({'instruction':instr},tokenizer=tokenizer,base=base) for instr in formatted_instrs]
        else:
            formatted_instrs = [format_message(fs+[{'instruction':instr}],tokenizer=tokenizer,base=base) for fs,instr in zip(cat_fs,formatted_instrs)]
    
        greedy_response = HF_generate(formatted_instrs,client,tokenizer,gen_kwargs,use_tgi=args.use_tgi,return_as_dict=False,return_probs=True,max_workers=len(formatted_instrs))
        # if args.mode != 'dola':
        if False:
        # if not args.mode:
            sampled_response = HF_generate(formatted_instrs,client,tokenizer,sample_kwargs,use_tgi=args.use_tgi,return_as_dict=False,return_probs=True,max_workers=len(formatted_instrs))
        else:
            sampled_response = None
        
        if args.use_tgi:
            sampled_response = [get_tgi_text(x) for x in sampled_response] if sampled_response is not None else None
        else:
            greedy_response = [g['text'] for g in greedy_response]
            sampled_response = [s['text'] for s in sampled_response] if sampled_response is not None else None
        
        if base:
            greedy_response = [clean_base_response(greedy_response) for greedy_response in greedy_response]
            sampled_response = [[clean_base_response(sr) for sr in sample_r] for sample_r in sampled_response] if sampled_response is not None else None
            
        if sampled_response is not None:
            for instr,g,s in zip(instrs,greedy_response,sampled_response):
                response_list.append({'instruction':instr,'greedy_response':g,'sampled_response':s})
        else:
            for instr,g in zip(instrs,greedy_response):
                response_list.append({'instruction':instr,'greedy_response':g})
        
    with open(response_path,'w') as f:
        for response in response_list:
            f.write(json.dumps(response)+'\n')

if __name__ == '__main__':
    main()