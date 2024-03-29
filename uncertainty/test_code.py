from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline,AutoModelForSequenceClassification,DebertaV2ForSequenceClassification,DebertaV2Tokenizer
import torch
from datasets import load_dataset,load_metric
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from multiprocessing import Pool
from collections import defaultdict
from copy import deepcopy
import pickle
import json
from tqdm import tqdm
import numpy as np
import re
import spacy
import random
import time
from huggingface_hub import InferenceClient
nli_model = 'potsawee/deberta-v3-large-mnli'
m = DebertaV2ForSequenceClassification.from_pretrained(nli_model).cuda()
tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)

text_a = 'I love to play basketball.'
text_b = 'I enjoy the sport of basketball alot, which is why i play it often.'

all_text = [(text_a,text_b) for _ in range(2)]

i = tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=all_text,
                    add_special_tokens=True, padding="longest",
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=True, return_attention_mask=True,
                )
inp  = i.input_ids

print (tokenizer.batch_decode(inp))
exit()

i = {k:v.cuda() for k,v in i.items()}
logits = m(**i).logits
p = torch.softmax(logits,dim=1).detach().cpu()
print (p)
exit()


# with open(path,'rb') as f:
#     data = pickle.load(f)

# d = data[0]
# for id,res in d['semantic_clusters'].items():
#     print ('entropy: ', d['all_cluster_entropies'][id])
#     print ('responses: ', res)
# exit()



m = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
instruction = "What is the highest mountain peak in Switzerland?"
c = "The highest mountain peak in Switzerland is Dufourspitze. It is located in the Pennine Alps, on the border between Switzerland and Italy, and has a height of 4,634 meters (15,203 feet) above sea level."
r = "The highest mountain peak in Switzerland is Dufourspitze. It stands at 4,634 meters (15,203 feet) tall and is located in the Alps on the border between Switzerland, Italy, and France."


nli_c = instruction + ' ' + c
nli_r = instruction + ' ' + r

nli_inp = nli_c + ' [SEP] ' + nli_r
reverse_nli_inp = nli_r + ' [SEP] ' + nli_c

tokenized_nli_inp = tokenizer([nli_inp,reverse_nli_inp],return_tensors='pt',padding='longest',truncation=True,max_length = 512)
tokenized_nli_inp = {k:v.cuda() for k,v in tokenized_nli_inp.items()}

with torch.no_grad():
    out = m(**tokenized_nli_inp).logits.detach().cpu()

pred_labels = torch.softmax(out,dim=1)
print (pred_labels)
exit()
    



ds = load_dataset('truthful_qa',"generation",split='validation')
topics = defaultdict(int)
ans_stats = []
question_len = []
for d in ds:
    topics[d['category']] += 1
    question_len.append(len(tokenizer.encode(d['question'])))
    ans_stats.extend([len(tokenizer.encode(t)) for t in d['correct_answers']])
    ans_stats.extend([len(tokenizer.encode(t)) for t in d['incorrect_answers']])
    
print (np.mean(question_len))
print (np.mean(ans_stats    ))
exit()

# config_names.remove('all')
# config_names = set(config_names)
val_dataset = load_dataset("cais/mmlu","all",split = 'validation')
test_dataset = load_dataset("cais/mmlu","all",split = 'test')
# dataset = load_dataset("truthful_qa","multiple_choice",split = 'validation')
print (val_dataset[0])
print (test_dataset[0])
exit()
topic_count = defaultdict(int)
non_topics = set()
for d in dataset:
    subject = d['subject']
    if subject in config_names:
        topic_count[subject] += 1
    else:
        non_topics.add(subject)

print (topic_count)
print (non_topics)
exit()

# print (dataset[0])
# exit()

client = InferenceClient(model = f"http://127.0.0.1:8083")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
messages = [
    {'role':'system','content':'i am smart'},
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]
formatted_text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
print (formatted_text)
exit()

out = client.text_generation(formatted_text,max_new_tokens=30,do_sample=False)

print (out)
exit()
# inp = f"### System:\n### User:\nHello, how are you doing?\n### Assistant:\n"

# out = client.text_generation(inp,max_new_tokens=5,details=True,best_of=2,do_sample=True)

# print ([t.logprob for t in out.details.tokens])
# print ([t.logprob for t in out.details.best_of_sequences[0].tokens])

# question_answerer = pipeline(
#             "question-answering", tokenizer="deepset/tinyroberta-squad2",
#             model="deepset/tinyroberta-squad2", framework="pt"
#         )

# questions = [{'text':'How are you?'} for _ in range(2)]
# context = [{'text':'I just ate chicken rice.'} for _ in range(2)]

# # process the whole batch
# out = question_answerer(question=[q['text'] for q in questions], context=[q['text'] for q in context])
# for q,o in zip(questions,out):
#     q['answer'] = o['answer']
#     q['score'] = o['score']
# print (questions)
# exit()


model = AutoModelForCausalLM.from_pretrained("Intel/neural-chat-7b-v3-3").cuda()
set_modules = set()
for n,m in model.named_modules():
    if isinstance(m,torch.nn.Linear):
        set_modules.add(n.split('.')[-1])
print (set_modules)


tokenizer = AutoTokenizer.from_pretrained("Intel/neural-chat-7b-v3-3")
exit()

inps = 'Hello, how are you doing?'
tokenized_inps = torch.tensor([tokenizer.encode(inps)]).to(model.device)
out = model.generate(tokenized_inps, max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
for decode_tok,logprob in zip(out.sequences[0],transition_scores[0]):
    print (tokenizer.decode(decode_tok),logprob)
    


