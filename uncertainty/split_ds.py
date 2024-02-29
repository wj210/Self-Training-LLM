import json
import random
import os
random.seed(42)
data_path = 'data/neural-chat-7b-v3-3_self_train.jsonl'
with open(data_path, 'r') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

random.shuffle(data)
test_size = 50
train_data = data[:-test_size]
test_data = data[-test_size:]

train_dir = 'data/train'
test_dir = 'data/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_path = f'{train_dir}/neural-chat-7b-v3-3_self_train.jsonl'
test_path = f'{test_dir}/neural-chat-7b-v3-3_self_val.jsonl'

with open(train_path,'w') as f:
    f.write('\n'.join([json.dumps(d) for d in train_data]))

with open(test_path,'w') as f:
    f.write('\n'.join([json.dumps(d) for d in test_data]))