max_epochs=2
max_steps=-1
bs=8
config_path="configs/model/tinyllama.yaml"

# if not using peft , unset use_peft, but would most likely using more than 1 gpu, then use accelerate below, set num process in deepspeed config and set $num_gpu flag
# accelerate launch --config_file configs/deepspeed.yaml --num_processes=$num_gpu uncertainty/train.py 

## SFT ###
python uncertainty/train.py \
--config_path $config_path \
--max_steps $max_steps \
--max_epochs $max_epochs \
--training_batch_size $bs \
--mode sft \
--save_last_only 


## DPO ##
question_filtering_threshold=0.5 # set tau_L to use consistency filtering S_L
unknown_threshold=0.8 # set tau_K for knowledge filtering (0.8 seems to be better, it depends on the model used. Look at the ablation study in the paper)
beta=0.3
max_epochs=-1
max_steps=300

python uncertainty/train.py \
--config_path $config_path \
--max_steps $max_steps \
--max_epochs $max_epochs \
--training_batch_size $bs \
--beta $beta \
--mode dpo \
--unknown_threshold $unknown_threshold \
--question_filtering \
--ref_as_chosen \