REVISION=main
max_input_length=3500
max_total_length=4096
port=8082 # set if using tgi
master_port=29488 # set if using tgi
mem_frac=0.7
num_seq=10 # set to 10 for sc-cot
model=Intel/neural-chat-7b-v3-3
# model=model_checkpoints/neural-chat-7b-v3-3_oracle_BSDetector # trained with self-gen labels
sharded=false
requests=320

export CUDA_VISIBLE_DEVICES=5,6
num_gpu=2
# if using local
text-generation-launcher --model-id $model --num-shard $num_gpu --port $port --max-input-length $max_input_length --master-port $master_port --cuda-memory-fraction $mem_frac --max-best-of $num_seq --sharded $sharded --max-total-tokens $max_total_length --disable-custom-kernels --max-concurrent-requests $requests \