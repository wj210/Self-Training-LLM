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
    parser.add_argument("--test_question_per_topic",  type = int,default =-1,help = 'if more than -1, we only test on this number of questions per topic')
    parser.add_argument("--extra_ds",  type = str,default =[],nargs = '+',help = 'test on datasets that are not trained on.')
    parser.add_argument("--filter_size",  type = float,default = 1.0,help = 'Top questions to take based on confidence/uncertainty')
    parser.add_argument("--openai_api_key_path",  type = str,default = '',help = 'a text file for openai api key, required only if using factscorer.')
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    with open(args.config_path,'r') as f: # model config
        config = SimpleNamespace(**yaml.safe_load(f))
    
    ds_config = {}
    if 'fixed' in args.topic_generator or 'predefined' in args.topic_generator or 'oracle' in args.topic_generator: # data config.
        ds_name = '_'.join(args.topic_generator.split('_')[1:]) 
    elif 'wiki' in args.topic_generator:
        ds_name = 'wiki'
    else:
        ds_name = 'self-generated'
    ds_config_path = f'configs/data/{ds_name}.yaml'
    with open(ds_config_path,'r') as f:
        ds_config = yaml.safe_load(f)
        
        
    ## Model path ##
    if args.scoring_method == 'BSDetector':
        scoring_name = 'conf' 
    elif args.scoring_method == 'semantic_consistency':
        scoring_name = 'entropy'
    else:
        scoring_name = 'hallu'
    model_dir = 'model_checkpoints'
    if not args.use_peft:
        model_dir += '_full'
        config.model_path = config.model_path.replace('model_checkpoints','model_checkpoints_full')
    
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
    
    ## Question path ## (to do testing on known dataset for catastrophic forgetting)
    # config.question_path = config.question_path.format(topic_generator=args.topic_generator)
    # with open(config.question_path,'rb') as f:
    #     question_ds = pickle.load(f)
    # known_ds = return_question_type(question_ds,args.scoring_method,'known') # TODO Do hallucination testing with this.
    
    ## Main Test dataset path ## (ignore for wiki)
    ds_config['test_dataset_path'] = ds_config['test_dataset_path'].format(dataset_name = ds_name,
                                                               test_qn_per_topic=args.test_question_per_topic)
    
    # load model and tokenizer
    test_load_path = config.model_path if args.trained else config.model_name
    tokenizer = load_tokenizer(test_load_path,padding_side='left') 
    if not args.trained:
        model_base_name = config.model_name.split('/')[-1].strip()
        config.result_path = os.path.dirname(config.result_path) + f'/{model_base_name}_pre.txt'
    
    os.makedirs(os.path.dirname(config.result_path),exist_ok=True)
        
    ## Type of data class
    ds_type = ds_config['answer_type']
    dataset_class = {'likelihood':LikelihoodDS,
                    'generation':GenerationDS}
    ds_kwargs = {'ds_name':ds_name}
    ds_kwargs['trained'] = args.trained
    
    with open(ds_config['test_dataset_path'],'r') as f: 
        test_dataset = [json.loads(line) for line in f]
    ## Few shot generation      
    test_num_fs = ds_config['test_few_shot']
    if isinstance(test_num_fs,int):
        if test_num_fs > 0:
            with open(config.fs_path,'rb') as f:
                test_fs_shot = pickle.load(f)
        else:
            test_fs_shot = []
    else:
        test_fs_shot = test_num_fs # either list or str
    
    ## FactScore.
    # if 'wiki' in args.topic_generator: # prepare data corpus for factscore if does not exist
    #     assert args.openai_api_key_path != '', 'Openai api key required for wiki data'
    #     db_path = '.cache/factscore/wiki.db'
    #     wiki_data_path  = '.cache/factscore/wiki.jsonl'
    #     knowledge_source_name = 'wiki'
        
    #     from factscore.factscorer import FactScorer
    #     factscorer = FactScorer(model_name = 'retrieval+llama+npm',
    #                              openai_key = args.openai_api_key_path
    #                              )
        
    #     # Check that the loaded DB includes all the documents in the test dataset, if not, we recreate the DB.
    #     if os.path.exists(db_path) and os.path.exists(wiki_data_path):
    #         with open(wiki_data_path,'r') as f:
    #             document_set = [json.loads(line) for line in f]
    #         required_documents = set([d['topic'] for d in test_dataset])
    #         if not required_documents.issubset(set([d['title'] for d in document_set])):
    #             os.remove(wiki_data_path) 
    #             os.remove(db_path)
        
    #     if not os.path.exists(db_path):
    #         print ('Setting up wiki database for FactScore')
    #         document_set = deepcopy(test_dataset)
    #         for d in document_set:
    #             d['text'] = d.pop('document')
    #             d['title'] = d.pop('topic')
    #             with open(wiki_data_path,'w') as f:
    #                 for t in document_set:
    #                     f.write(json.dumps(t,ensure_ascii=False)+'\n')
    #         factscorer.register_knowledge_source(knowledge_source_name,
    #                                               data_path = wiki_data_path,
    #                                               db_path = db_path)
    #         factscorer = FactScorer(model_name = 'retrieval+llama+npm',
    #                              openai_key = args.openai_api_key_path
    #                              )
            
    #         print (f'Done setting up wiki database for FactScore!')
    #     ds_kwargs['knowledge_source'] = knowledge_source_name
    #     ds_kwargs['factscorer'] = factscorer
    
    test_ds = dataset_class[ds_type](test_dataset,tokenizer,config.model_name,few_shots=test_fs_shot,kwargs=ds_kwargs)
    gen_kwargs = ds_config['gen_kwargs']
    all_testing_dict = {ds_name:{'ds':test_ds,
                                    'gen_kwargs':gen_kwargs,
                                    'ds_type':ds_type,
                                    }}
    
    if len(args.extra_ds) > 0:
        for e_ds in args.extra_ds:
            e_config_path = f'configs/data/{e_ds}.yaml'
            with open(e_config_path,'r') as f:
                extra_ds_config = yaml.safe_load(f)
            extra_test_ds,extra_test_fs = load_test_ds(extra_ds_config)
            e_ds_type = extra_ds_config['answer_type']
            ds_kwargs['ds_name'] = e_ds
            extra_test_ds = [vars(d) for d in extra_test_ds] # convert to dict
            extra_test_ds = dataset_class[e_ds_type](extra_test_ds,tokenizer,config.model_name,few_shots=extra_test_fs,kwargs=ds_kwargs)
            all_testing_dict[e_ds] = {'ds':extra_test_ds,
                                        'gen_kwargs':extra_ds_config['gen_kwargs'],
                                        'ds_type':e_ds_type}

    ## Conduct testing for each ds ## 
    logged_results = {}
    for test_ds_name,ds_dict in all_testing_dict.items():
        if (args.use_tgi and ds_dict['ds_type'] == 'generation'): # Load model according to the type of dataset.
            model = InferenceClient(model = f"http://127.0.0.1:{args.port}")
        else:
            model = load_hf_model(test_load_path,quantized=args.quantized).eval() 
        
        if test_ds_name == 'wiki' and ds_dict['ds_type'] == 'generation': # only compute consistency score for generation ds.
            scorer = NLIScorer(model,config.model_name,tokenizer,args.scoring_method,1.0,max_response_tokens=ds_config.get('max_response_tokens',128),answer_generator=args.answer_generator,answer_generator_port=-1,ref_as_chosen=False)
            ds_dict['ds'].scorer = scorer
        
        result_dict = eval_fixed_ds(ds_dict['ds'],
                        model,
                        config.model_name,
                        tokenizer,
                        test_ds_name,
                        args.test_batch_size,
                        ds_type = ds_dict['ds_type'],
                        num_samples = 10,
                        gen_kwargs = ds_dict['gen_kwargs'],
                        trained = args.trained,
                        use_tgi = args.use_tgi,
                        )
        logged_results[test_ds_name] = result_dict
    
    # Log Test results #
    with open(config.result_path,'w') as f:
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
            for topic,acc in result_dict['topic_acc'].items():
                f.write(f'{topic} Acc: {acc:.3f}\n')
            f.write('*'*120+'\n\n')

if __name__ == '__main__':
    main()