REVISION=main
max_input_length=3500
max_total_length=4096
port=8082 # port to connect to.
master_port=29488 
mem_frac=0.7
num_seq=10 # set to 10 for num sequences
model=mistralai/Mistral-7B-Instruct-v0.2 # change the model name here.
# model=mistralai/Mistral-7B-v0.1
# model=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# model=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
sharded=false
requests=320
# export CUDA_VISIBLE_DEVICES=2 # set if need to use specific gpu
num_gpu=1
# if using local
text-generation-launcher --model-id $model --num-shard $num_gpu --port $port --max-input-length $max_input_length --master-port $master_port --cuda-memory-fraction $mem_frac --max-best-of $num_seq --sharded $sharded --max-total-tokens $max_total_length --disable-custom-kernels --max-concurrent-requests $requests \