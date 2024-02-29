# Run the main script
echo "Running job with CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
cuda_path="cuda_dev/cuda_visible_devices.txt"
if [ -f $cuda_path ]; then
  export CUDA_VISIBLE_DEVICES=$(cat $cuda_path)
  num_gpu=$(cat "$cuda_path" | tr ', ' '\n' | grep -c '[0-9]')
  echo "num gpus = $num_gpu"
else
  echo "cuda_visible_devices.txt file not found."
fi

export ACCELERATE_LOG_LEVEL=info
export WANDB_API_KEY='4d0cfb6b964e4092b544eaa50ffa07ae36cc5249'

google_custom_api="AIzaSyDQZMXbf9IpVzzFzNFxRoiIyQFMKzQQKnc"
search_engine_id="627d798e7078b41bc"
export GOOGLE_CUSTOM_SEARCH_URL="https://www.googleapis.com/customsearch/v1?key=$google_custom_api&cx=$search_engine_id&q=" # **Blur out beside uploading to github

answer_generator="gpt4"
scoring_method="BSDetector"
training=true
testing=true

accelerate launch --config_file uncertainty/configs/deepspeed.yaml --num_processes=$num_gpu uncertainty/main.py \
--answer_generator $answer_generator \
--scoring_method $scoring_method \
--training $training \
--testing $testing 


# answer_generator="self"
# accelerate launch --config_file uncertainty/configs/deepspeed.yaml --num_processes=$num_gpu uncertainty/main.py \
# --answer_generator $answer_generator \
# --scoring_method $scoring_method \
# --training $training \
# --testing $testing 


# answer_generator="oracle"
# accelerate launch --config_file uncertainty/configs/deepspeed.yaml --num_processes=$num_gpu uncertainty/main.py \
# --answer_generator $answer_generator \
# --scoring_method $scoring_method \
# --training $training \
# --testing $testing 
# --use_tgi true