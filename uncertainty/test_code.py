import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM
from huggingface_hub import InferenceClient
from datasets import load_dataset
from templates import question_generation_examples
import pickle
from tqdm import tqdm
import spacy
import torch
from utils import load_tokenizer
import random
tokenizer = load_tokenizer("model_checkpoints_full/SFT/tinyllama_ultrachat")
# tokenizer = load_tokenizer("model_checkpoints_full/wiki/zephyr_hallu_self_top100")

client = InferenceClient(model = "http://127.0.0.1:8088")
def check_question(question):
    if 'based on the' in question and ('document' in question or 'information' in question or 'text' in question):
        return False
    return True

def clean_question(question):
    if '?' in question:
        question = question.split('?')[0].strip()+ '?'
    return question

def prompt_fn(topic,document=''):
    # return f"Document: {document}\n\nInstruction: propose one question regarding \"{topic}\". The question should be created based on details in the document.\nQ: "
    return f"{document}\n\nInstruction: propose only one question regarding \"{topic}\".The question should be detailed and specific.\n\nProposed Question:" 

# def prompt_fn_answer(document,qn):
#     if document != '':
#         return f"{document}\n\nQuestion: {qn}\nAnswer: "
#     else:
#         return f"Question: {qn}\nAnswer: "
    
def prompt_fn_answer(document,qn):
    if document != '':
        return f"Evidence: {document}\n\nBased on evidence from Wikipedia, {qn}\nAnswer: "
    else:
        return f"Question: {qn}\nAnswer: "    

topic='Doja Cat'
document= "\"Mooo!\" (often stylized in all caps as \"MOOO!\") is a song by American rapper and singer Doja Cat. After originally being self-published exclusively as a music video on August 10, 2018, it became a viral internet meme and amassed over 578 million views. It was subsequently released as the lead single from the deluxe edition (and third overall) of her debut studio album Amala. The viral success of \"Mooo!\" is considered to have been a major influence to Doja Cat's internet fame, ultimately \"setting the tone for her career\", despite being considered by Doja Cat herself as a \"throwaway\" and a \"joke\".\n\nBackground and recording \nPrior to the release of \"Mooo!\", Doja Cat had released her \"moderately successful\" debut studio album, Amala, in March 2018. She developed the song as an inside joke alongside her fans in early August 2018, not expecting it to go further than SoundCloud. She told Dazed, \"We started it on Instagram live, just me and 60 other people, and we all had fun coming up with puns and metaphors.\" The song was inspired by Doja Cat's cow-print costume set which she wears throughout the song's music video. She wrote and recorded the song in six hours, while in bed in the costume. Doja Cat used a sample of Wes Montgomery's \"Polka Dots and Moonbeams\", which producer Troy NōKA had chopped and sent to her the night before. After making a beat with the sample and recording vocals in Logic Pro, she immediately began filming the song's music video from her bedroom. According to Doja Cat, she completed the song and its video within 12.5 hours of one day.\n\nThe song's music video gained over five million views in two weeks. After the video's viral success an updated single was released.\n\nComposition\n\n\"Mooo!\" is \"a rather simple, jazzy song about the important things in life: eating cheeseburgers, maybe doing some kissing, and generally not being in the mood to do anything else.\" A novelty song, Doja Cat raps about being a cow, despite her name, and the pleasures of farm life in a pseudo-sexual way. The song contains a plethora of cows \"mooing\" (mainly COW – SINGLE MOO, ANIMAL 02 from Sound Ideas's The General Series 6000) background vocals over \"swelling harmonies\" and jazz guitar. The refrain goes \"Bitch I'm a cow / Bitch I'm a cow / I'm not a cat / I don't say meow\". The song features a lyrical reference to the nursery rhyme \"Old MacDonald Had a Farm (E-I-E-I-O!)\", while also referencing hip hop songs including Ludacris's \"Move Bitch\", Schoolboy Q's \"Collard Greens\", Chamillionaire's \"Ridin'\", Kelis's \"Milkshake\", Tear Da Club Up Thugs's \"Slob on My Nob\", and Wu Tang Clan's \"C.R.E.A.M.\".\n\nMusic video\n\nTo prepare for filming, Doja Cat hammered a green bed sheet to her bedroom wall to act as a green screen and inserted GIFs from Google into Photo Booth. The video for \"Mooo!\" features Doja Cat clad in cow-print pajamas with french fries in her nose and eating various fast food items. She raps in front of a green screen which alternates between cartoonish GIFs of food, farms, and bouncing anime breasts, as well as brief video samples from Cyriak's \"cows & cows & cows\". The video was filmed and edited by Doja Cat herself in the timespan of five hours maximum. She said in an interview that the green screen was actually made of her childhood bedsheets, as she was \"obsessed with green\" as a kid. The DIY video has been praised for its \"lo-fi\", and \"low budget\" nature. Sofia Mele of Billboard compared the video to that of John Mayer's \"New Light\" while describing it as a \"meme-maker's paradise, charmingly kitschy in its use of green screen\".\n\nThe American animal rights organization PETA responded to the song's music video with a parody video told from the perspective of a cow that Kristin Corry of Vice described as \"pretty damn rude\". Doja Cat responded to the parody, saying, \"PETA can't say shit and they can suck it because I didn't actually hurt anybody. I didn't hurt any cows, dogs, cats, or frogs, or fucking ants. I'm not worth picking on.\" In addition to the parody, Doja Cat also responded to the negative criticism towards \"Mooo!\", tweeting: \"I love that the majority of you guys are healthy and normal and then all of the people who don't like moo are taking their lives and a song I wrote about cows all too seriously, losing hair over it."
text = [{'role':'user','content':prompt_fn(topic,document)}]
qn_system_prompt = [{'role':'system','content':'You are a student who is eager to learn about new things. You are to form a question that you lack knowledge in.'}]
all_fs = []
all_fs_w_ans = []
all_fs_w_ans_nd = []
for fs in question_generation_examples[:3]:
    all_fs.extend([{'role':'user','content':prompt_fn(fs['topic'],document = fs['document'])},
               {'role':'assistant','content':fs['question']}])
    all_fs_w_ans.extend([{'role':'user','content':prompt_fn_answer(fs['document'],fs['question'])},
               {'role':'assistant','content':fs['answer']}])
    all_fs_w_ans_nd.extend([{'role':'user','content':prompt_fn_answer('',fs['question'])},
               {'role':'assistant','content':fs['answer']}])
text =  qn_system_prompt + all_fs + text

input_text = tokenizer.apply_chat_template(text,tokenize=False,add_generation_prompt=True)
kwargs  = {'max_new_tokens':64, 'do_sample':False, 'repetition_penalty':1.1}
ref_kwargs = {'max_new_tokens':256, 'do_sample':False,'repetition_penalty':1.1}
sample_kwargs = {'max_new_tokens':256, 'do_sample':True, 'temperature':0.5,'repetition_penalty':1.1,'best_of':4,'details':True}
# out = client.text_generation(input_text, **kwargs)
# qn = clean_question(out)
qn = "What inspired Doja Cat to create the viral hit \"Mooo!\"?"
ref_qn = [{'role':'user','content':prompt_fn_answer(document,qn)}]
# print (qn)
ref_qn = all_fs_w_ans + ref_qn
ref_qn = tokenizer.apply_chat_template(ref_qn,tokenize=False,add_generation_prompt=True)
ref_ans = client.text_generation(ref_qn, **ref_kwargs)
sample_ans = client.text_generation(ref_qn, **sample_kwargs)
sampled_text = [sample_ans.generated_text] + [ss.generated_text for ss in sample_ans.details.best_of_sequences]
print (ref_ans)
for s in sampled_text:
    print ('sampled:',s)
exit()

sample_qn = [{'role':'user','content':prompt_fn_answer('',qn)}]
# sample_qn =  sample_qn
sample_qn = tokenizer.apply_chat_template(sample_qn,tokenize=False,add_generation_prompt=True)
sample_ans = client.text_generation(sample_qn, **ref_kwargs)
print (sample_ans)
exit()
sampled_text = [sample_ans.generated_text] + [ss.generated_text for ss in sample_ans.details.best_of_sequences]
for s in sampled_text:
    print ('sampled:',s)
