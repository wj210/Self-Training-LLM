from datasets import load_dataset,get_dataset_config_names
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from utils import format_response,get_extract_template,if_instruction_tuned,HF_generate
from collections import defaultdict
from copy import deepcopy
from time import time


def defaultint():
    return defaultdict(int)

def compute_ece(predictions, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE).
    Parameters:
    - predictions: List of tuples (confidence, outcome), where confidence is the predicted probability
      of the positive class, and outcome is the actual class label (0 or 1).
    - n_bins: Number of bins to use for grouping confidence scores.
    
    Returns:
    - ECE: The expected calibration error.
    """
    confidences, outcomes = zip(*predictions)
    confidences = np.array(confidences)
    outcomes = np.array(outcomes)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        # Identify the indices of samples in the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_confidences = confidences[in_bin]
            bin_outcomes = outcomes[in_bin]
            
            # Compute the accuracy and average confidence for the bin
            bin_accuracy = np.mean(bin_outcomes)
            bin_confidence = np.mean(bin_confidences)
            
            # Update ECE
            bin_weight = np.mean(in_bin)
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
    
    return ece

def eval_fixed_ds(ds,model,model_name,tokenizer,ds_name,batch_size,ds_type= 'likelihood',num_samples=10,gen_kwargs=None,trained=False,use_tgi=False):
    """
    Given a ds, where each sample is a QA_sample, init a Dataset and perform eval based on highest likelihood choice.
    """
    if trained or if_instruction_tuned(model_name):
        instruction_tuned = True
    else:
        instruction_tuned = False
    extract_ans_fn = get_extract_template(model_name,instruction_tuned) if ds_name == 'truthful_qa' else None
    test_dl = DataLoader(ds,batch_size=batch_size,collate_fn=ds.collate_fn,num_workers=8,drop_last=False)
    acc = []
    confidence_score_tracker = []
    topic_acc = defaultdict(list)
    hallucination_score = []
    
    # Set up sampling kwargs if getting hallucination score
    sample_kwargs = deepcopy(gen_kwargs)
    sample_kwargs['do_sample'] = True
    sample_kwargs['temperature'] = 1.0
    if not use_tgi:
        sample_kwargs['num_return_sequences'] = num_samples  
    else:
        sample_kwargs['best_of'] = num_samples
        sample_kwargs['details'] = True
        gen_kwargs['details'] = True
    
    fs_logger = []
    if ds_type == 'likelihood':
        pred_dict = defaultdict(defaultint)
        answer_dict = defaultdict(int)
    
    ## Eval on subject dataset ## (full dataset)
    for i,batch in tqdm(enumerate(test_dl),total = len(test_dl),desc = f'Running eval on {ds_name}'):
        input_ids = batch['input_ids']
        answer = batch['answer']
        topics = batch['topic']
        if ds_type == 'likelihood':
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                logits = model(input_ids).logits.detach().cpu()
            ds.derive_prediction(logits,batch,pred_dict,answer_dict)
        elif ds_type == 'generation':
            pred = HF_generate(input_ids,model,tokenizer,gen_kwargs,use_tgi=use_tgi,return_as_dict=False,return_probs=True,max_workers=len(input_ids))
            if use_tgi:
                pred_text = [p.generated_text for p in pred]
            else:
                pred_text = [p['text'] for p in pred]
            if ds.scorer is not None:
                sampled_pred = HF_generate(input_ids,model,tokenizer,sample_kwargs,use_tgi=use_tgi,return_as_dict=False,return_probs=True,max_workers=len(input_ids))
            else:
                sampled_pred = None
            for i,(p,a,topic) in enumerate(zip(pred_text,answer,topics)):
                if 'wiki' not in ds_name:
                    acc_score = ds.score_prediction(p,a,topic,extract_ans_fn)
                    acc.append(acc_score)
                if sampled_pred is not None:
                    hallu_score = ds.score_hallucination(input_ids[i],pred[i],sampled_pred[i])
                    if hallu_score:
                        hallucination_score.append(hallu_score)
            if 'wiki' in ds_name: 
                fs_score_dict = ds.score_prediction(pred_text,None,topics) # compute entire batch
                acc_score = fs_score_dict['score']
                for topic,decisions in zip(fs_score_dict['topics'],fs_score_dict['decisions']):
                    log_ = {}
                    log_['topic'] = topic
                    log_['document'] = ds.topic2docu.get(topic,'None')
                    log_['results'] = []
                    for x in decisions:
                        atom = x['atom']
                        supported = x['is_supported']
                        log_['results'].append(f'Fact: {atom}. Supported: {supported}'+'\n')
                    fs_logger.append(log_)
                acc.append(acc_score)
                
        else:
            raise NotImplementedError

    if ds_type == 'likelihood':
        for sample_id,sample in pred_dict.items():
            greedy_choice = sorted(sample.items(),key = lambda x: x[1],reverse=True)[0][0]
            if greedy_choice == answer_dict[sample_id]:
                acc.append(1)
            else:
                acc.append(0)
    
    ## Compute final stats
    result_dict = {}
    result_dict['acc'] = sum(acc)/len(acc)
    if len(topic_acc) > 0:
        result_dict['topic_acc'] = {k:sum(v)/len(v) for k,v in topic_acc.items()}
    
    ## Confidence and ECE
    if len(confidence_score_tracker) > 0:
        ece_bins = 10
        result_dict['ece'] = compute_ece(confidence_score_tracker,ece_bins)
        result_dict['conf_score'] = np.mean([c[0] for c in confidence_score_tracker]).item()

    if len(hallucination_score) > 0:
        if ds.scorer.scoring_method == 'BSDetector':
            result_dict['hallucination_score'] = 1. - np.mean(hallucination_score).item() # since confidence, take complement
        else:
            result_dict['hallucination_score'] = np.mean(hallucination_score).item()
    if len(fs_logger) > 0:
        result_dict['fs_logs'] = fs_logger
    
    return result_dict
