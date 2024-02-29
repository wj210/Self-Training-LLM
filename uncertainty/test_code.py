from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch

# client = InferenceClient(model = f"http://127.0.0.1:8082")


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
    


