num_gpus=4
export WANDB_DISABLED="true"
export OMP_NUM_THREADS=12
torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" preference_model_dist.py \
  --fp16 False \
  --bf16 False \
  --model_name_or_path "google/flan-t5-large" \
  --cache_dir "/nlp/scr/kawin/.cache" \
  --output_dir "/nlp/scr/kawin/models/flan-t5-large-rerun" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --save_strategy "no" \
  --learning_rate 5e-5 \
  --evaluation_strategy "epoch" \
  --logging_strategy "steps" \
  --logging_steps 100 \
  --overwrite_output_dir 
