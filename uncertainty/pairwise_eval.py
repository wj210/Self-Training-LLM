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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/model/tinyllama.yaml")
    parser.add_argument("--base_path", type=str, default="responses/sft/tinyllama_self.jsonl")
    parser.add_argument("--mode", type=str, default="sft")
    parser.add_argument("--question_filtering", action='store_true')
    parser.add_argument("--multiple_pref",  action = 'store_true',help= 'Create multiple preference samples per instruction')
    parser.add_argument("--unknown_threshold",  default = 0.5,help= 'score to select the split unknown samples')
    parser.add_argument("--answer_generator", type=str, default="self")
    parser.add_argument("--scoring_method", type=str, default="SelfCheckGPT")
    parser.add_argument("--beta",  default = 0.1,help= 'beta value for DPO')
    parser.add_argument("--openai_api_key_path",  type = str,default = '',help = 'a text file for openai api key, required only if using factscorer.')
    parser.add_argument("--known",  action = 'store_true',help= 'test on known samples (training set)')
    args = parser.parse_args()
    seed_all(42)
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    
    assert args.openai_api_key_path != '' , 'OpenAI API key path is required for LLMJUDGE'
    
    with open(args.openai_api_key_path,'r') as f:
        openai_key = f.readline().strip()
    os.environ["OPENAI_API_KEY"] = openai_key
    
    path_flag = ''
    if args.question_filtering:
        path_flag += '_qf'
    if args.multiple_pref:
        path_flag += '_multi'
        
    if args.mode == 'dpo':
        path_flag += f'_{args.unknown_threshold}'
        path_flag += f'_{args.beta}beta'

    if 'tinyllama' in args.base_path:
        base_model_name = 'tinyllama'
    elif 'llama2_7b' in args.base_path:
        base_model_name = 'llama2_7b'
    elif 'llama2_13b' in args.base_path:
        base_model_name = 'llama2_13b'
    elif 'test' in args.base_path:
        base_model_name = 'gpt3.5'
    
    tokenizer = load_tokenizer(config.base_model_name)
        
    config.result_path = config.result_path.format(mode=args.mode,answer_generator=args.answer_generator)  
    
    config.result_path = config.result_path.replace('.txt',f'{path_flag}.txt')
    
    if args.known:
        config.result_path = config.result_path.replace('.txt','_known.txt')
        args.base_path = args.base_path.replace('.jsonl','_known.jsonl')
        test_ds_path = f'data/wiki/{base_model_name}_known.jsonl'
    else:
        test_ds_path = 'data/wiki/test.jsonl'
        
    eval_path = config.result_path.replace('test_results','responses').replace('.txt','.jsonl')
    
    assert os.path.exists(args.base_path), 'Base responses not found'
    assert eval_path != args.base_path, 'Base and eval paths are not different!'
    
    os.makedirs(os.path.dirname(config.result_path),exist_ok=True)
    
    scorer = NLIScorer(None,None,None,args.scoring_method,1.0,max_response_tokens=128,answer_generator=args.answer_generator,ref_as_chosen=False,use_tgi = True)
    llm_judge = LLMJudge(engine = "gpt-4o")

    with open(test_ds_path,'r') as f:
        test_ds = [json.loads(l) for l in f.readlines()]
    
    instr_2_dict = {d['instruction']:d['document'] for d in test_ds}
    
    with open(args.base_path,'r') as f:
        base_response_dict = [json.loads(l) for l in f.readlines()]
    
    eval_model_name = args.config_path.split('/')[-1].split('.')[0]
    base_key = 'greedy_response'

    if args.mode != 'oracle':
        with open(eval_path,'r') as f:
            eval_response_dict = [json.loads(l) for l in f.readlines()]
    else:
        eval_response_dict = test_ds # change keys
            
    eval_instr_to_response = {d['instruction']:d['greedy_response'] for d in eval_response_dict}
    
    if True:
        judge_dict = []
        for d in base_response_dict:
            instr = d['instruction']
            judge_dict.append({'instruction':instr,'base_response':d[base_key],'post_response':eval_instr_to_response[instr],'document':instr_2_dict[instr]})
        
        results,cost,no_count = llm_judge.evaluate(judge_dict)
        
        tie_count = sum([1 if r == 'tie' else 0 for r in results])
        win_count = sum([1 if r == 'win' else 0 for r in results])
        lose_count = sum([1 if r == 'lose' else 0 for r in results])
        
        total_counted = len(results)
        
        tie_score = tie_count/total_counted
        win_score = win_count/total_counted
        lose_score = lose_count/total_counted
    # else:
    #     win_score = 0
    #     tie_score = 0
    #     lose_score = 0
    #     cost = 0
    #     no_count = 0
        
    score_key = SCORE_KEY[args.scoring_method]
    base_hallu,eval_hallu = [],[]
    # if args.mode != 'dola':
    if False:
        for bd,ed in tqdm(zip(base_response_dict,eval_response_dict),total = len(base_response_dict),desc = 'Hallucination'):
            instr = bd['instruction']
            base_hallu.append(scorer.get_score(instr,bd['greedy_response'],bd['sampled_response'])[score_key])
            eval_hallu.append(scorer.get_score(instr,ed['greedy_response'],ed['sampled_response'])[score_key])
    else:
        base_hallu = [0]
        eval_hallu = [0]
    
    base_hallu = sum(base_hallu)/len(base_hallu)
    eval_hallu = sum(eval_hallu)/len(eval_hallu)
    
    base_gen_len = np.mean([len(tokenizer.encode(d[base_key])) for d in base_response_dict])
    eval_gen_len = np.mean([len(tokenizer.encode(d['greedy_response'])) for d in eval_response_dict])
    
    
    print (f'Win: {win_score:.3f}, Tie: {tie_score:.3f}, Lose: {lose_score:.3f}, Cost: {cost:.3f}, No Count: {no_count}')
    print (f"Base hallu score: {base_hallu:.3f}, Eval hallu score: {eval_hallu:.3f}")
    with open(config.result_path,'a') as f:
        f.write(f'\n************ Base model: {base_model_name}, Eval model: {eval_model_name}{path_flag} ************\n')
        f.write(f'Win: {win_score:.3f}\n')
        f.write(f'Tie: {tie_score:.3f}\n')
        f.write(f'Lose: {lose_score:.3f}\n')
        f.write(f'Cost: {cost:.3f}\n')
        f.write(f'No Count: {no_count}\n')
        f.write(f"Base hallu score: {base_hallu:.3f}\n")
        f.write(f"Eval hallu score: {eval_hallu:.3f}\n")
        f.write(f"Base gen len: {base_gen_len:.3f}\n")
        f.write(f"Eval gen len: {eval_gen_len:.3f}\n")

if __name__ == '__main__':
    main()
    
    
    
    
    
    