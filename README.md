# Self-Learning-llm

Code adapted from https://github.com/teddy-f-47/self-learning-llm-public

Main code in `uncertainty`, scripts to run are in `script`

**Important files**
- `wiki_ques_gen.sh`- creates dataset
- `train.sh` - train either based on ds saved under '/train' folder
- `test.sh` - do testing
- `tgi.sh` - setup the model api to do fast-inference (recommended)

**Requirements**
- run `pip install -r requirements.txt`
- Follow https://github.com/google-research/bleurt to install bleurt to evaluate truthful_qa
- if factscore is not installed correctly, follow https://github.com/shmsw25/FActScore
- If using TGI - install by following https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#local-install


**Data Generation**

`wiki_ques_gen.sh` does the following:

1) Sample topics based on `num_topics` and generate `questions_per_topic` qn per topic, given the wiki document relating to each topic 
2) Check for duplicates based on rouge and split into train and test set.
3) On the train set, generate both reference(greedy decoded and given supporting document) and `num_samples` sampled responses w/o the document
4) Compute a form of hallucination score based on `scoring_method` set.
5) Split into known and unknown questions and save them.
6) Generate the dpo training set only on unknown questions, the 'chosen labels' are generated based on `answer_generator` set, default uses the reference as chosen, if set gpt4 or gpt3.5, will use them to generate labels in place of the reference generated above. In theory, the self-generated answer should be good enough since the model is given additional context as help while the 'rejected' is chosen from the set of sampled responses based on hallucination score, ie the response which contradicts the chosen the most is set as rejected.

**Training**

- `train.sh` can train using either PEFT or full parameter training , set the `use_peft` flag. parameters set in `configs/training/lora.yaml`
- Dpo or sft in `configs/training`
- If multi_gpu for full parameter, in `configs/deepspeed.yaml`
- model config is in `configs/model`

**Testing**

run `script/test.sh`, the parameters are specified inside.

**Faster Inference**
- The code uses TGI for for either data generation or testing. The inference is much faster than standard way of loading model and doing batch generation with `model.generate`.
- The only troublesome part is that the model have to be first loaded, by running `tgi.sh` and then running the main script. So if we want to do testing with 2 different models, we have to setup first model -> testing -> unset first model and set 2nd -> testing.
- `tgi.sh` basically sets up the model on your local hardware for you to make API calls (similar to making it via OpenAI). The code is setup to do multi-threading to increase inference speed.


