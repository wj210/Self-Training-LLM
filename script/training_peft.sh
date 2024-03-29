topic_generator='wiki'
max_epochs=2 # set either epoch or steps, not both, the unset one will be -1
max_steps=-1
bs=8
scoring_method='SelfCheckGPT'
config_path="configs/model/mistral_instruct.yaml"
answer_generator=self
filter_size=1.0 # lower if want to do filter filtering for more unknown samples
# training_args_path="configs/training/sft_trainer.yaml" # if using SFT unset this

# if not using peft , unset use_peft, but would most likely using more than 1 gpu, then use accelerate below, set num process in deepspeed config and set $num_gpu flag
# accelerate launch --config_file configs/deepspeed.yaml --num_processes=$num_gpu uncertainty/train.py 

python uncertainty/train.py \
--config_path $config_path \
--max_steps $max_steps \
--max_epochs $max_epochs \
--training_batch_size $bs \
--topic_generator $topic_generator \
--answer_generator $answer_generator \
--scoring_method $scoring_method \
--filter_size $filter_size \
--use_peft true \
# --training_args_path $training_args_path # if using SFT else default DPO
