#!/bin/bash

# export MODEL_PATH="THUDM/CogVideoX-2b"
export MODEL_PATH="/eva_data5/kuoyuhuan/VideoGenAI/cache/hub/models--THUDM--CogVideoX-2B/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
VIDEO_ROOT_DIR="/eva_data5/kuoyuhuan/DLP_final/data"
CSV_PATH="/eva_data5/kuoyuhuan/DLP_final/prompt/video_prompts.csv"


# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config.yaml --multi_gpu \
  train_controlnet.py \
  --tracker_name "cogvideox-controlnet" \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --validation_prompt "car is going in the ocean, beautiful waves:::ship in the vulcano" \
  --validation_video "../resources/car.mp4:::../resources/ship.mp4" \
  --validation_prompt_separator ::: \
  --num_inference_steps 28 \
  --num_validation_videos 1 \
  --validation_steps 500 \
  --seed 42 \
  --mixed_precision fp16 \
  --output_dir "results" \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_root_dir $VIDEO_ROOT_DIR \
  --csv_path $CSV_PATH \
  --stride_min 1 \
  --stride_max 3 \
  --hflip_p 0.5 \
  --controlnet_type "canny" \
  --controlnet_transformer_num_layers 4 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 0.5 \
  --init_from_transformer \
  --train_batch_size 1 \
  --dataloader_num_workers 0 \
  --num_train_epochs 1 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --use_8bit_adam 
  # --report_to wandb
  # --pretrained_controlnet_path "cogvideox-controlnet-2b/checkpoint-2000.pt" \
    