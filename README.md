# Self-Learning-llm

Main code in `uncertainty`, scripts to run are in `script`

**Important files**
- `wiki_ques_gen.sh`- generates both SFT and DPO dataset as well as performing knowledge filtering
- `train.sh` - SFT and DPO
- `test.sh` - do eval on the primary test set on Wikipedia
- `tgi.sh` - setup the model api to do fast-inference (recommended)

**Requirements**
- run `pip install -r requirements.txt`
- If using TGI - install by following https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#local-install


**Data Generation**

`wiki_ques_gen.sh` does the following:

1) Generate questions using GPT3.5 on the predefined list of wikipedia articles from https://huggingface.co/datasets/wikimedia/wikipedia in `titles.py`. Both for test and train (step 1 in paper figure)
2) Generate the greedy decoded response, $y_c^*$.
3) Generate the K sampled responses given the document ($Y_c$) as context for consistency filtering and compute the consistency score, $S_L$.
4) Generate K sampled responses without the context for $Y_r$ and compute knowledge score $S_K$.

**Training**

- `train.sh` can train using either PEFT or full parameter training , set the `use_peft` flag. parameters set in `configs/training/lora.yaml`
- DPO or SFT in `configs/training`
- If multi_gpu for full parameter, in `configs/deepspeed.yaml`
- model config is in `configs/model`

**Testing**
run `script/test.sh`, the parameters are specified inside.

* Note that you should first generate the response of the baseline model, being the SFT model, $G_{SFT}$.
`script/test.sh` first generates the response and then perform pairwise ranking with the base_response, which is set in the `base_path` argument.

**Faster Inference**
- The code uses TGI for for either data generation or testing. The inference is much faster than standard way of loading model and doing batch generation with `model.generate`.
- The only troublesome part is that the model have to be first loaded, by running `tgi.sh` and then running the main script. So if we want to do testing with 2 different models, we have to set up first model -> testing -> unset first model and set 2nd -> testing.
- `tgi.sh` basically sets up the model on your local hardware for you to make API calls (similar to making it via OpenAI). The code is set up to do multi-threading to increase inference speed.

**Extra Notes**
- To work with other LLMs, you can just change or replicate the config format in `configs/model`.
- This work could potentionally work with other forms of unstructured knowledge source besides Wikipedia. The main processing code to gather the documents is the `get_predefined_topics` function in `topic_generator.py` file. As long as the entries in the data generation contains the field `document`, it will construct the dataset.



