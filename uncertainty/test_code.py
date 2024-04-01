import os
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
a = [{'role':'user','content':'What is the capital of France?'}]

print (tokenizer.apply_chat_template(a,tokenize=False,add_generation_prompt=True))
exit()


path = "data/wiki/train/mistralinstr_hallu_self.jsonl"
new_path = "data/wiki/train/mistralinstr_hallu_self_1.jsonl"
with open(path, 'r') as f:
    ds = [json.loads(line) for line in f]

new_d = []
qn_correction= 0 
ans_correction = 0


def check_answer(ans):
    if 'there is no' in ans:
        if 'mention' in ans or 'record' in ans or 'information' in ans or 'data' in ans or 'evidence' in ans:
            return True
    if 'i apologize' in ans or 'i cannot answer' in ans:
        return True
    return False

for d in ds:
    instr = d['instruction']
    if '\n' in instr.split('?')[-1]:
        d['instruction'] = instr.split('?')[0].strip() + '?'
        qn_correction += 1
    chosen_ans = d['chosen_ans']
    if check_answer(chosen_ans):
        ans_correction += 1
        continue
    new_d.append(d)

with open(new_path, 'w') as f:
    for d in new_d:
        f.write(json.dumps(d,ensure_ascii=False) + '\n')

print (qn_correction, ans_correction)