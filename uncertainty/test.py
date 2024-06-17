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
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082,help = 'port for TGI')
    parser.add_argument("--config_path", type=str, default="",required=True)
    parser.add_argument("--answer_generator", type=str, default="self")
    parser.add_argument("--scoring_method", type=str, default="BSDetector")
    parser.add_argument("--use_tgi", action = 'store_true',help = 'use TGI for loaded model to do eval')
    parser.add_argument("--test_batch_size",  type = int,default = 4)
    parser.add_argument("--quantized",  action = 'store_true',help = 'quantized model for inference')
    parser.add_argument("--num_samples", type=int, default=5,help = 'number of sampled responses')
    parser.add_argument("--test_datasets",  type = str,default =[],nargs = '+',help = 'test on datasets that are not trained on.')
    parser.add_argument("--openai_api_key_path",  type = str,default = '',help = 'a text file for openai api key, required only if using factscorer.')
    parser.add_argument("--question_filtering", action='store_true')
    parser.add_argument("--multiple_pref",  action = 'store_true',help= 'Create multiple preference samples per instruction')
    parser.add_argument("--unknown_threshold",  default = 0.5,help= 'score to select the split unknown samples')
    parser.add_argument("--beta",  default = 0.1,help= 'beta value for DPO')
    parser.add_argument("--dola",  action = 'store_true',help= 'To use decoding contrastive DOLA, only for base model')
    parser.add_argument("--checkpoint",  default = '',type = str)
    parser.add_argument("--mode",  default = 'dpo',type = str)
    args = parser.parse_args()
    ## Seed ## 
    seed_all(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    ## Model path ##
    path_flag = ''
    if args.question_filtering:
        path_flag += '_qf'
    if args.multiple_pref:
        path_flag += '_multi'
    
    
    if args.dola:
        assert not args.use_tgi and args.mode == 'base', 'DOLA only for base model, not supported for TGI'
    
    ## Result path ##
    config.result_path = config.result_path.format(answer_generator=args.answer_generator,
                                                mode=args.mode,
                                                )
    ## Model path ##
    if args.mode == 'dpo':
        path_flag += f'_{args.unknown_threshold}'
        model_path = config.dpo_model_path.format(answer_generator=args.answer_generator) + path_flag
        model_path = model_path + f'_{args.beta}beta'
        config.result_path = config.result_path.replace('.txt',f'{path_flag}.txt')
        if args.checkpoint != '':
            model_path = os.path.join(model_path,args.checkpoint)
    elif args.mode == 'sft':
        model_path = config.sft_model_path.format(answer_generator=args.answer_generator) + path_flag
        config.result_path = config.result_path.replace('.txt',f'{path_flag}.txt')
    elif args.mode == 'base':
        model_path = config.base_model_name
        config.result_path = config.result_path.replace(f"_{args.answer_generator}","")
        if args.dola:
            config.result_path = config.result_path.replace('.txt','_dola.txt')
    else:
        raise ValueError('Invalid mode')

    # load model and tokenizer
    tokenizer = load_tokenizer(model_path,padding_side='left') 
    if args.mode == 'base' and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
            test_path = ds_config['test_dataset_path'].format(dataset_name = 'wiki')
        else:
            test_path = ''
        
        ## Load test path ##
        if ds_name == 'cf': # test for catestrophic forgetting
            config.answer_path = config.dpo_answer_path.format(answer_generator = args.answer_generator).replace('.pkl',f'{path_flag}.pkl')
            with open(config.answer_path,'rb') as f:
                question_ds = pickle.load(f)
            test_ds = return_question_type(question_ds,args.scoring_method,'known',args.unknown_threshold) 
            test_fs = []
        else:
            test_ds,test_fs = load_test_ds(ds_config,test_path,max_test_samples=-1)
            
        e_ds_type = ds_config['answer_type']
        e_gen_kwargs = ds_config['gen_kwargs']
        if args.dola: ## Added support for dola
            e_gen_kwargs['repetition_penalty'] = 1.2
            e_gen_kwargs['dola_layer'] = [18,20,22,24,26,28,30,32] if 'tinyllama' not in model_path.lower() else 'high'
            
        ds_kwargs['ds_name'] = ds_name
        if not isinstance(test_ds[0],dict):
            test_ds = [vars(d) for d in test_ds] # convert to dict
        if 'wiki' in ds_name: # prepare data corpus for factscore if does not exist
            assert args.openai_api_key_path != '', 'Openai api key required for wiki data'
            knowledge_name = 'wiki'
            
            db_path = f'.cache/factscore/{knowledge_name}.db'
            wiki_data_path  = f'.cache/factscore/{knowledge_name}.jsonl'
            
            from factscore.factscorer import FactScorer
            factscorer = FactScorer(
                                    model_name = "retrieval+llama+npm",
                                    # relevance_model_name = "ChatGPT",
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
                factscorer.register_knowledge_source(name = knowledge_name,
                                                    data_path = wiki_data_path,
                                                    db_path = db_path)
                factscorer = FactScorer(
                                    model_name = "retrieval+llama+npm",
                                    # relevance_model_name = "ChatGPT",
                                    openai_key = args.openai_api_key_path,
                                    max_passage_length = 256 ## set max passage retrival length
                                    )
                
                print (f'Done setting up wiki database for FactScore!')
            topic2docu = {d['topic']:d['document'] for d in test_ds}
            ds_kwargs['knowledge_source'] = knowledge_name
            ds_kwargs['factscorer'] = factscorer
            ds_kwargs['topic2docu'] = topic2docu

        test_ds = dataset_class[e_ds_type](test_ds,tokenizer,model_path,few_shots=test_fs,kwargs=ds_kwargs)
        all_testing_dict[ds_name] = {'ds':test_ds,
                                    'gen_kwargs':e_gen_kwargs,
                                    'ds_type':e_ds_type,
                                    }
    # Load model if not using TGI else use TGI API
    if args.use_tgi: 
        model = InferenceClient(model = f"http://127.0.0.1:{args.port}")
    else:
        model = load_hf_model(model_path,quantized=args.quantized).eval()

    ## Conduct testing for each ds ## 
    logged_results = {}
    for test_ds_name,ds_dict in all_testing_dict.items():
        if ds_dict['ds_type'] == 'generation': # only compute consistency score for generation ds.
            scorer = NLIScorer(model,model_path,tokenizer,args.scoring_method,1.0,max_response_tokens=128,answer_generator=args.answer_generator,answer_generator_port=-1,ref_as_chosen=False,use_tgi = args.use_tgi)
            ds_dict['ds'].scorer = scorer
        ## Check for existing result ##
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
        result_dict = eval_fixed_ds(ds_dict['ds'],
                        model,
                        tokenizer,
                        args.test_batch_size,
                        ds_type = ds_dict['ds_type'],
                        num_samples = args.num_samples,
                        gen_kwargs = ds_dict['gen_kwargs'],
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
            if 'avg_length' in result_dict:
                f.write(f'Average length : {result_dict["avg_length"]:.3f}\n')
                f.write(f'Num correct facts: {result_dict["correct_facts"]:.3f}\n')
                f.write(f'Num wrong facts: {result_dict["wrong_facts"]:.3f}\n')
            if 'irrelevance_score' in result_dict:
                f.write(f'irrelevance: {result_dict["irrelevance_score"]:.3f}\n')
            f.write('*'*120+'\n\n')
    
    ## Logging for factscore ##
    for test_ds_name,result_dict in logged_results.items():
        if test_ds_name == 'wiki':
            if 'fs_logs' in result_dict:
                test_result_base_dir = os.path.dirname(config.result_path).replace('test_results','log_results')
                os.makedirs(test_result_base_dir,exist_ok=True)
                log_file = config.result_path.split('/')[-1].strip()
                log_path = os.path.join(test_result_base_dir,log_file)
                with open(log_path,'w') as f:
                    for log in result_dict['fs_logs']:
                        topic = log['topic']
                        generation = log['generation']
                        result = log['results']
                        instr = log['instruction'].split('<|user|>')[1].split('</s>')[0].strip()
                        f.write('*'*50 + f' Topic: {topic} ' + '*'*50 + '\n')
                        f.write(f'Instruction: {instr}\n')
                        f.write(f'Generation: {generation}\n')
                        f.write(f'Score: {log["score"]:.2f}\n')
                        f.write('*'*100 + '\n')
                        f.write('Facts:\n')
                        for res in result:
                            f.write(res)
                        f.write('*'*120+'\n\n')
    
if __name__ == '__main__':
    main()