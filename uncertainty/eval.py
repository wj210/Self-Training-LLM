from datasets import load_dataset,get_dataset_config_names
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from utils import format_response,get_extract_template,if_instruction_tuned,HF_generate
from data_utils import LikelihoodDS,GenerationDS
from collections import defaultdict
from copy import deepcopy
from time import time
import json

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
        
    ## Eval on subject dataset ## (full dataset)
    for batch in tqdm(test_dl,total = len(test_dl),desc = f'Running eval on {ds_name}'):
        input_ids = batch['input_ids']
        answer = batch['answer']
        topics = batch['topic']
        if ds_type == 'likelihood':
            choices = batch['choices']
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                logits = model(input_ids).logits
            for i,logit in enumerate(logits):
                conf_score = []
                pred_dict = (ds.derive_prediction(logit[-1],choices[i],gen_kwargs['temperature'],num_samples)) # take last logit
                greedy = pred_dict['greedy']
                acc_score = int(greedy == answer[i])
                acc.append(acc_score)
                topic_acc[topics[i]].append(acc_score)
                sampled = pred_dict['sampled']
                for sample in sampled:
                    if sample.lower() == greedy.lower():
                        conf_score.append(1)
                    else:
                        conf_score.append(0)                
                conf_score = np.mean(conf_score)
                confidence_score_tracker.append((conf_score,acc_score)) # tuple of confidence and correct/wrong.
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
                if ds_name != 'wiki':
                    acc_score = ds.score_prediction(p,a,topic,extract_ans_fn)
                    topic_acc[topic].append(acc_score)
                    acc.append(acc_score)
                if sampled_pred is not None:
                    hallucination_score.append(ds.score_hallucination(input_ids[i],pred[i],sampled_pred[i]))
            if ds_name == 'wiki':
                acc.append(0.) ##TODO find a way to work with FactScore with wiki.
                continue
                fs_score_dict = ds.score_prediction(pred_text,None,topics) # compute entire batch
                acc_score = fs_score_dict['score']
                # with open('testing.txt','w') as f:
                #     for x in sum(fs_score_dict['decisions'],[]):
                #         atom = x['atom']
                #         supported = x['is_supported']
                #         f.write(f'Fact: {atom}. Supported: {supported}'+'\n')
                acc.append(acc_score)
                
        else:
            raise NotImplementedError
    
    ## Compute final stats
    result_dict = {}
    result_dict['acc'] = sum(acc)/len(acc)
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
    
    return result_dict
