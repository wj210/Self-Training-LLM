bs=10
config_path="configs/model/tinyllama.yaml"
openai_api_key=openai_api_key # Must set
port=8082
mode=dpo
base_path="responses/sft/tinyllama_self.jsonl" # change accordingly to type of model.
unknown_threshold=0.8

# if use tgi set use_tgi and port, else unset both. 

## Note that the base path is the path to be compared to, which is the resposnes from the SFT model. IF it is not yet generated, run the below code.

## Generate response for SFT
# python uncertainty/generate_response.py \
# --config_path $config_path \
# --port $port \
# --use_tgi \
# --mode sft \
# --batch_size $bs \

python uncertainty/generate_response.py \
--config_path $config_path \
--port $port \
--use_tgi \
--mode $mode \
--batch_size $bs \
--unknown_threshold $unknown_threshold \
--question_filtering

python uncertainty/pairwise_eval.py \
--config_path $config_path \
--answer_generator $answer_generator \
--mode $mode \
--openai_api_key_path $openai_api_key \
--beta $beta \
--unknown_threshold $unknown_threshold \
--base_path $base_path \
--question_filtering