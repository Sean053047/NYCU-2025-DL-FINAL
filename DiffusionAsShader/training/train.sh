export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="THUDM/CogVideoX-2B"
DATA_ROOT="/eva_data5/kuoyuhuan/DLP_final/data"
CKPT_PATH="/eva_data5/kuoyuhuan/DLP_final/DiffusionAsShader/training/v1/checkpoint-1000"

accelerate launch --config_file accelerate_config.yaml \
    cogvideox_image_to_video_sft.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir v1\
    --dataset_file $DATA_ROOT/metadata.csv \
    --video_column "cam_video" \
    --tracking_column "lidar_video" \
    --caption_column "prompt" \
    --height 480 \
    --width 720 \
    --num_tracking_blocks 18 \
    --load_tensors \
    --mixed_precision fp16 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --checkpointing_steps 500 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --lr_scheduler constant \
    --enable_slicing \
    --enable_tiling \
    --resume_from_checkpoint latest
    # --precompute_embeddings