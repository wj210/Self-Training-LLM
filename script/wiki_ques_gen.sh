num_samples=10
num_topics=1000
port=8082
test_size=10
questions_per_topic=8
config_path="configs/model/tinyllama.yaml"
openai_api_key_path='openai_api_key.txt' # set your own key

python uncertainty/wiki_generation.py \
--config_path $config_path \
--ref_as_chosen \
--num_samples $num_samples \
--port $port \
--num_topics $num_topics \
--use_tgi \
--test_size $test_size \
--openai_api_key_path $openai_api_key_path \
--questions_per_topic $questions_per_topic \

