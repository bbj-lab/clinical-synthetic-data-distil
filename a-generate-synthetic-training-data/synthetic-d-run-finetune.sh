accelerate launch --config_file ../configs/deep_speed2.yaml --multi_gpu --num_machines 1  --num_processes 8  finetune.py --min_difficulty "1" --input_file all --model_id meta-llama/Llama-3.2-1B 2>&1 | tee ./logs/log-1B.txt

