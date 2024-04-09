import os
import json
import argparse
import yaml
from types import SimpleNamespace
from huggingface_hub import InferenceClient
from scorer import NLIScorer
from utils import *
from data_utils import *
from eval import eval_fixed_ds
from copy import deepcopy
import torch
import random
import numpy as np



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082,help = 'port for TGI')
    parser.add_argument("--mode", type=str, default="data_generation")
    parser.add_argument("--config_path", type=str, default="",required=True)
    parser.add_argument("--answer_generator", type=str, default="self")
    parser.add_argument("--scoring_method", type=str, default="BSDetector")
    parser.add_argument("--topic_generator", type=str, default="fixed_truthful_qa")
    parser.add_argument("--use_tgi", type=bool, default=False,help = 'use TGI for loaded model to do eval')
    parser.add_argument("--use_peft", type=bool, default=False,help = 'use peft for training')
    parser.add_argument("--test_batch_size",  type = int,default = 4)
    parser.add_argument("--quantized",  type = bool,default = False,help = 'quantized model for inference')
    parser.add_argument("--trained",  type = bool,default = False,help = 'if trained, load model path else model_name')
    parser.add_argument("--num_samples", type=int, default=5,help = 'number of sampled responses')
    parser.add_argument("--test_datasets",  type = str,default =[],nargs = '+',help = 'test on datasets that are not trained on.')
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--openai_api_key_path",  type = str,default = '',help = 'a text file for openai api key, required only if using factscorer.')
    args = parser.parse_args()
    ## Seed ## 
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    ## Model path ##
    if args.scoring_method == 'BSDetector':
        scoring_name = 'conf' 
    elif args.scoring_method == 'semantic_consistency':
        scoring_name = 'entropy'
    else:
        scoring_name = 'hallu'
    ## Model path ##
    config.model_path = config.model_path.format(topic_generator=args.topic_generator,
                                                answer_generator=args.answer_generator,
                                                scoring_name=scoring_name,
                                                filter_size = int(100*args.filter_size))
    
    ## Result path ##
    config.result_path = config.result_path.format(topic_generator=args.topic_generator,
                                                answer_generator=args.answer_generator,
                                                scoring_name=scoring_name,
                                                filter_size = int(100*args.filter_size))
    if not args.use_peft:
        config.model_path = config.model_path.replace('model_checkpoints','model_checkpoints_full')
        config.result_path = config.result_path.split('txt')[0] + '_full.txt'
    
    # load model and tokenizer
    test_load_path = config.model_path if args.trained else config.model_name
    tokenizer = load_tokenizer(test_load_path) 
    if not args.trained:
        model_base_name = config.model_name.split('/')[-1].strip()
        config.result_path = 'test_results/base' + f'/{model_base_name}.txt'
    
    os.makedirs(os.path.dirname(config.result_path),exist_ok=True)
        
    ## Type of data class
    
    dataset_class = {'likelihood':LikelihoodDS,
                    'generation':GenerationDS}

    all_testing_dict = {}
    
    for ds_name in args.test_datasets:
        ds_kwargs = {}
        if 'wiki' in ds_name or ds_name == 'cf':
            config_path = f'configs/data/wiki.yaml'
        else:
            config_path = f'configs/data/{ds_name}.yaml'
        with open(config_path,'r') as f:
            ds_config = yaml.safe_load(f)
        if 'wiki' in ds_name:
            test_path = ds_config['test_dataset_path'].format(dataset_name = args.topic_generator)
        else:
            test_path = ''
        
        if ds_name == 'cf': # test for catestrophic forgetting
            config.question_path = config.question_path.format(topic_generator=args.topic_generator)
            with open(config.question_path,'rb') as f:
                question_ds = pickle.load(f)
            test_ds = return_question_type(question_ds,args.scoring_method,'known') 
            test_fs = []
        else:
            test_ds,test_fs = load_test_ds(ds_config,test_path,max_test_samples=100) # eval 100 first.
        e_ds_type = ds_config['answer_type']
        e_gen_kwargs = ds_config['gen_kwargs']
        ds_kwargs['ds_name'] = ds_name
        ds_kwargs['trained'] = args.trained
        if not isinstance(test_ds[0],dict):
            test_ds = [vars(d) for d in test_ds] # convert to dict

        if 'wiki' in ds_name: # prepare data corpus for factscore if does not exist
            assert args.openai_api_key_path != '', 'Openai api key required for wiki data'
            db_path = f'.cache/factscore/{args.topic_generator}.db'
            wiki_data_path  = f'.cache/factscore/{args.topic_generator}.jsonl'
            
            from factscore.factscorer import FactScorer
            factscorer = FactScorer(
                                    # model_name = 'retrieval+ChatGPT',
                                    model_name = "retrieval+llama+npm",
                                    openai_key = args.openai_api_key_path,
                                    max_passage_length = 256 ## set max passage retrival length
                                    )
            
            # Check that the loaded DB includes all the documents in the test dataset, if not, we recreate the DB.
            if os.path.exists(db_path) and os.path.exists(wiki_data_path):
                with open(wiki_data_path,'r') as f:
                    document_set = [json.loads(line) for line in f]
                required_documents = set([d['topic'] for d in test_ds])
                if not required_documents.issubset(set([d['title'] for d in document_set])):
                    os.remove(wiki_data_path) 
                    os.remove(db_path)
            
            if not os.path.exists(db_path):
                print ('Setting up wiki database for FactScore')
                document_set = deepcopy(test_ds)
                for d in document_set:
                    d['text'] = d.pop('document')
                    d['title'] = d.pop('topic')
                    with open(wiki_data_path,'w') as f:
                        for t in document_set:
                            f.write(json.dumps(t,ensure_ascii=False)+'\n')
                factscorer.register_knowledge_source(name = args.topic_generator,
                                                    data_path = wiki_data_path,
                                                    db_path = db_path)
                factscorer = FactScorer(
                                    # model_name = 'retrieval+ChatGPT',
                                    model_name = "retrieval+llama+npm",
                                    openai_key = args.openai_api_key_path,
                                    max_passage_length = 256 ## set max passage retrival length
                                    )
                
                print (f'Done setting up wiki database for FactScore!')
            topic2docu = {d['topic']:d['document'] for d in test_ds}
            ds_kwargs['knowledge_source'] = args.topic_generator
            ds_kwargs['factscorer'] = factscorer
            ds_kwargs['topic2docu'] = topic2docu

        test_ds = dataset_class[e_ds_type](test_ds,tokenizer,config.model_name,few_shots=test_fs,kwargs=ds_kwargs)
        all_testing_dict[ds_name] = {'ds':test_ds,
                                    'gen_kwargs':e_gen_kwargs,
                                    'ds_type':e_ds_type,
                                    }
    # Load model if not using TGI else use TGI API
    if args.use_tgi: 
        model = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    else:
        model = load_hf_model(test_load_path,quantized=args.quantized).eval()

    ## Conduct testing for each ds ## 
    logged_results = {}
    for test_ds_name,ds_dict in all_testing_dict.items():
        skipped = False
        if os.path.exists(config.result_path):
            with open(config.result_path,'r') as f:
                for line in f:
                    if f'Dataset: {test_ds_name}' in line:
                        print (f'{test_ds_name} already tested, skipping, to override delete the line.')
                        skipped = True
                        break
        if skipped:
            continue
        if ds_dict['ds_type'] == 'generation': # only compute consistency score for generation ds.
            scorer = NLIScorer(model,config.model_name,tokenizer,args.scoring_method,1.0,max_response_tokens=128,answer_generator=args.answer_generator,answer_generator_port=-1,ref_as_chosen=False)
            ds_dict['ds'].scorer = scorer
            tokenizer.padding_side = 'left'
        result_dict = eval_fixed_ds(ds_dict['ds'],
                        model,
                        config.model_name,
                        tokenizer,
                        test_ds_name,
                        args.test_batch_size,
                        ds_type = ds_dict['ds_type'],
                        num_samples = args.num_samples,
                        gen_kwargs = ds_dict['gen_kwargs'],
                        trained = args.trained,
                        use_tgi = args.use_tgi,
                        )
        logged_results[test_ds_name] = result_dict
    
    # Log Test results #
    with open(config.result_path,'a') as f:
        for test_ds_name,result_dict in logged_results.items():
            f.write('*'*50 + f' Dataset: {test_ds_name} ' + '*'*50 + '\n')
            f.write(f'Acc: {result_dict["acc"]:.3f}\n')
            if 'conf_score' in result_dict:
                f.write(f'Conf Score: {result_dict["conf_score"]:.3f}\n')
            if 'ece' in result_dict:
                f.write(f'ECE: {result_dict["ece"]:3f}\n')
            if 'hallucination_score' in result_dict:
                f.write(f'Hallucination Score: {result_dict["hallucination_score"]:.3f}\n')
            f.write(f'--------------Topic-wise accuracy--------------\n')
            if 'topic_acc' in result_dict:
                for topic,acc in result_dict['topic_acc'].items():
                    f.write(f'{topic} Acc: {acc:.3f}\n')
            f.write('*'*120+'\n\n')
    
    ## Logging for factscore ##
    for test_ds_name,result_dict in logged_results.items():
        if 'fs_logs' in result_dict:
            test_result_base_dir = os.path.dirname(config.result_path).replace('test_results','log_results')
            os.makedirs(test_result_base_dir,exist_ok=True)
            log_file = config.result_path.split('/')[-1].strip()
            log_path = os.path.join(test_result_base_dir,log_file)
            with open(log_path,'w') as f:
                for log in result_dict['fs_logs']:
                    topic = log['topic']
                    document = log['document']
                    result = log['results']
                    f.write('*'*50 + f' Topic: {topic} ' + '*'*50 + '\n')
                    f.write(f'Document: {document}\n')
                    f.write('Facts:\n')
                    for res in result:
                        f.write(res)
                    f.write('*'*120+'\n\n')
    
if __name__ == '__main__':
    main()