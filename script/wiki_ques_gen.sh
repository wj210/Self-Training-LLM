topic_generator=wiki
num_samples=10
questions_per_topic=2 # set num qn per topic
num_topics=2000 
# scoring_method='semantic_consistency' # choose the scoring methods
# scoring_method='BSDetector'
scoring_method='SelfCheckGPT'
port=8082 # set port to the same port as in tgi.sh if using tgi , remember set --use_tgi flag.
test_size=500 # num of test samples to hold out
config_path="configs/model/mistral_instruct.yaml" # using mistral-instruct
answer_generator=self # answer is self-generated.

python uncertainty/wiki_generation.py \
--config_path $config_path \
--questions_per_topic $questions_per_topic \
--num_samples $num_samples \
--topic_generator $topic_generator \
--answer_generator $answer_generator \
--scoring_method $scoring_method \
--port $port \
--num_topics $num_topics \
--use_tgi \
--test_size $test_size
