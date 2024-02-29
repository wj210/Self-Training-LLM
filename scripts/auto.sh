#!/bin/bash

## setup the code parameters

module load cuda11.7/toolkit
module load cuda11.7/blas

export OMP_NUM_THREADS=8
# The required free memory in MiB
REQUIRED_MEMORY=39000  # For example, 70 GB
REQUIRED_GPUS=1       # Number of GPUs needed

p=PA100q
w=node03
# p=RTXA6Kq
# w=node09
c=4 # num cpus

# This array will hold the PIDs of the Python sub-scripts
OCCUPY_SCRIPT_PIDS=()
USED_GPUS=()

# Define a function to cleanup background processes
cleanup() {
    echo "Keyboard interrupt received. Cleaning up..."
    # Kill the Python sub-scripts
    for pid in "${OCCUPY_SCRIPT_PIDS[@]}"; do
        kill $pid
    done
    echo "Cleanup done. Exiting."
    exit
}

# Trap the SIGINT signal (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

allocate_gpu_memory() {
  while true; do
    # Reset GPU list
    gpu_id=0

    # Run the command and write the output to a temporary file
    srun -p $p -w $w nvidia-smi | grep -E "[0-9]+MiB / [0-9]+MiB" > gpu_memory_info_w14.txt

    # Read from the temporary file
    while IFS= read -r line; do 
      if [[ " ${USED_GPUS[@]} " =~ " ${gpu_id} " ]]; then
          ((gpu_id++))
          continue
        fi

      # Extract used and total memory
      used_memory=$(echo $line | awk '{print $9}' | sed 's/MiB//')
      total_memory=$(echo $line | awk '{print $11}' | sed 's/MiB//')

      # Calculate actual free memory
      actual_free_memory=$((total_memory - used_memory))

      # Check if the free memory is greater than or equal to the required memory
      if [ "$actual_free_memory" -ge "$REQUIRED_MEMORY" ]; then
        # Run the Python sub-script to occupy memory on this GPU
        srun -p $p -w $w --exact --job-name=t_$gpu_id python mem.py --device_no $gpu_id --memory $REQUIRED_MEMORY &
        OCCUPY_SCRIPT_PIDS+=($!)
        echo "Started process $! to occupy GPU $gpu_id"
        USED_GPUS+=("$gpu_id")
      fi

      if [ ${#USED_GPUS[@]} -ge $REQUIRED_GPUS ]; then
        break  # Break the while loop if the condition is met
      fi

      # Increment GPU ID counter
      ((gpu_id++))
    done < gpu_memory_info_w14.txt

    # Clean up the temporary file
    rm gpu_memory_info_w14.txt

    # Check if the required number of GPUs is met
    if [ ${#USED_GPUS[@]} -ge $REQUIRED_GPUS ]; then
      echo "Found ${#USED_GPUS[@]} GPUs with enough memory: ${USED_GPUS[*]}"
      # Kill the Python sub-scripts
      # echo "rest very long..."
      # sleep 10000000000
      for pid in "${OCCUPY_SCRIPT_PIDS[@]}"; do
        kill $pid
      done
      OCCUPY_SCRIPT_PIDS=()
      break  # Break the while loop if the condition is met
    else
      echo "Not enough GPUs found, left gpus to find: $((REQUIRED_GPUS - ${#USED_GPUS[@]}))"
      sleep 20
    fi
  done
  # Set CUDA_VISIBLE_DEVICES to the GPUs found
  SELECTED_GPUS=("${USED_GPUS[@]:0:$REQUIRED_GPUS}")
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${SELECTED_GPUS[*]}")
  echo $CUDA_VISIBLE_DEVICES > cuda_dev/cuda_visible_devices.txt
  # export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
}

allocate_gpu_memory

srun -p $p -w $w -c $c --verbose --job-name=self_learning --gpus=$num_gpu bash scripts/run_peft.sh

trap cleanup SIGINT

# Check for success.txt
if [ ! -f success.txt ]; then
    echo "Error in main script. Occupying all GPUs indefinitely."
    while true; do
      for gpu_id in "${USED_GPUS[@]}"; do
          srun -p $p -w $w --gpus=1 python mem.py --device_no $gpu_id --memory $REQUIRED_MEMORY --device_no $gpu_id &
          OCCUPY_SCRIPT_PIDS+=($!)
      done
      sleep 100000000
    done
fi



