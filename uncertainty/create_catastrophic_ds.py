import os
import json
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import torch
import random
import argparse
import yaml
import pickle
from utils import load_hf_model,load_tokenizer
from data_utils import get_fixed_ds,LikelihoodDS,GenerationDS
from torch.utils.data import DataLoader
from types import SimpleNamespace
from dataclasses import asdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default="configs/model/mistral.yaml")
    parser.add_argument("--batch_size",  type = int,default = 4)
    parser.add_argument("--ds_name",  type = str,default = 'mmlu')
    parser.add_argument("--test_qn_per_topic",  type = int,default = 50)
    parser.add_argument("--quantized",  type = bool,default = False,help = 'quantized model for inference')
    args = parser.parse_args()
    
    ## Seeding
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    model_config = yaml.load(open(args.model_config_path,'r'),Loader=yaml.FullLoader)
    model_name = model_config['model_name']
    cs_path = model_config['catastrophic_ds_path'].format(ds_name=args.ds_name)
    cs_dir = os.path.dirname(cs_path)
    os.makedirs(cs_dir,exist_ok=True)
    
    
    ds_config_path = f'configs/data/{args.ds_name}.yaml'
    ds_config = yaml.load(open(ds_config_path,'r'),Loader=yaml.FullLoader)
    
    model = load_hf_model(model_name,args.quantized)
    model.eval()
    tokenizer = load_tokenizer(model_name,padding_side = 'left')
    ## Preload test-ds
    if 'mmlu' in ds_config['ds_name']:
        test_ds_path = ds_config['test_dataset_path'].format(dataset_name=args.ds_name,test_qn_per_topic=args.test_qn_per_topic)
    else:
        test_ds_path = ds_config['test_dataset_path'].format(dataset_name=args.ds_name)
    
    ## Preload few-shot examples
    fs_path = ds_config['fs_path'].format(dataset_name=args.ds_name)
    if os.path.exists(fs_path):
        few_shots = pickle.load(open(fs_path,'rb'))
        few_shots = {k:[SimpleNamespace(**fs) for fs in v] for k,v in few_shots.items()}

    if os.path.exists(test_ds_path):
        with open(test_ds_path,'r') as f:
            test_ds = [json.loads(line) for line in f]
            test_ds = [SimpleNamespace(**x) for x in test_ds]
    else:
        test_ds,few_shots,_ = get_fixed_ds(ds_config_path,0,args.test_qn_per_topic)
        os.makedirs(os.path.dirname(test_ds_path),exist_ok=True)
        ## Save if not exist
        with open(test_ds_path,'w') as f:
            for sample in test_ds:
                f.write(json.dumps(asdict(sample),ensure_ascii=False) + '\n')
        with open(fs_path,'wb') as f:
            for fs_topic,fs in few_shots.items():
                few_shots[fs_topic] = [asdict(x) for x in fs]
            pickle.dump(few_shots,f)
    
    ds_type = ds_config['answer_type']
    assert ds_type == 'likelihood', 'So far only support likelihood dataset for catastrophic eval'
    gen_kwargs = ds_config['gen_kwargs']
    
    # Begin eval here
    cs_ds = [] # record down the samples that are correct.
    dataset_class = {'likelihood':LikelihoodDS,
                     'generation':GenerationDS}
    test_ds = dataset_class[ds_type](test_ds,tokenizer,model_name,few_shots=few_shots,return_original_question=True)# rmb set return to True
    test_dl = DataLoader(test_ds,batch_size=args.batch_size,collate_fn=test_ds.collate_fn,num_workers=8,drop_last=False)
    for batch in tqdm(test_dl,total = len(test_dl),desc = f'Recording correct samples on {args.ds_name}'):
        input_ids = batch['input_ids'].to(model.device)
        answer = batch['answer']
        choices = batch['choices']
        data_sample = batch['data_sample']
        with torch.no_grad():
            logits = model(input_ids).logits.detach().cpu()
        
        for i,logit in enumerate(logits):
            pred_dict = (test_ds.derive_prediction(logit[-1],choices[i],gen_kwargs['temperature'],1)) # take last logit
            greedy = pred_dict['greedy']
            if greedy == answer[i]:
                cs_ds.append(vars(data_sample[i]))
    
    with open(cs_path,'w') as f:
        for sample in cs_ds:
            f.write(json.dumps(sample,ensure_ascii=False) + '\n') 

if __name__ == "__main__":
    main()
    
    
