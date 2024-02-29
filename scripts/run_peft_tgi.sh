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

# srun -p $p -w $w -c $c --verbose --job-name=self_learning --gpus=$num_gpu python experiment_training.py
# srun -p $p -w $w -c $c --verbose --job-name=self_learning --gpus=$num_gpu python experiment_post_training.py


google_custom_api="AIzaSyDQZMXbf9IpVzzFzNFxRoiIyQFMKzQQKnc"
search_engine_id="627d798e7078b41bc"
export GOOGLE_CUSTOM_SEARCH_URL="https://www.googleapis.com/customsearch/v1?key=$google_custom_api&cx=$search_engine_id&q=" # **Blur out beside uploading to github


scoring_method="semantic_consistency"
use_peft=true

## Training
for ans_generator in self gpt4 oracle
do
  python uncertainty/main.py \
  --answer_generator $ans_generator \
  --scoring_method $scoring_method \
  --training true \
  --use_peft $use_peft \
  --num_iterations 200 \
  --port 8082
done

## Testing with tgi
# for ans_generator in gpt4
# do
#   python uncertainty/main.py \
#   --answer_generator $ans_generator \
#   --scoring_method $scoring_method \
#   --testing true \
#   --use_peft $use_peft \
#   --num_iterations 200 \
#   --use_tgi true \
#   --port 8083
# done