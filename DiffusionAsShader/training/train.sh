export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="THUDM/CogVideoX-2B"
DATA_ROOT="/eva_data5/kuoyuhuan/DLP_final/data"

accelerate launch --config_file accelerate_config.yaml \
    cogvideox_image_to_video_sft.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir cogvideox-sft\
    --dataset_file $DATA_ROOT/metadata.csv \
    --video_column "cam_video" \
    --tracking_column "cam_video" \
    --caption_column "prompt" \
    --height 480 \
    --width 720 \
    --num_tracking_blocks 18 \
    --load_tensors \
    --mixed_precision fp16 \
    --train_batch_size 1 \
    --num_train_epochs 10 \
    --checkpointing_steps 1000 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --lr_scheduler constant \
    --enable_slicing \
    --enable_tiling 