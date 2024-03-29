topic_generator='wiki'
filter_size=1.0
num_samples=10
bs=8
test_question_per_topic=50 # can ignore for wiki
scoring_method='SelfCheckGPT'
answer_generator=self
config_path="configs/model/mistral_instruct.yaml"
port=8082 # if using tgi, set to the same port as in tgi.sh
openai_api_key=openai_api_key.txt # create a openai key path
extra_ds=truthful_qa # set any extra datasets other than the one used to train the model specified in topic_generator. ie train on wiki, test on both wiki and truthful_qa

# if trained using peft, set use_peft to point to correct model checkpoint,
# if testing trained model, set --trained else remove it to evaluate base_model
# quantized to 4-bit to save mem --quantized

python uncertainty/test.py \
--config_path $config_path \
--use_tgi true \
--test_batch_size $bs \
--filter_size $filter_size \
--trained true \
--topic_generator $topic_generator \
--answer_generator $answer_generator \
--scoring_method $scoring_method \
--quantized true \
--test_question_per_topic $test_question_per_topic \
--use_peft true \
--port $port \
--openai_api_key_path $openai_api_key \
--extra_ds $extra_ds
    
