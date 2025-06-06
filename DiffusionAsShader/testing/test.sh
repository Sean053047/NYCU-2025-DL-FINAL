set -e -x

export CUDA_VISIBLE_DEVICES=0
VID_NAME=("CAMERA_LEFT_BACK_00080.mp4"
          "CAMERA_LEFT_BACK_00070.mp4"
          "CAMERA_LEFT_BACK_00030.mp4"
          "CAMERA_LEFT_FRONT_00080.mp4"
          "CAMERA_LEFT_FRONT_00070.mp4"
          "CAMERA_LEFT_FRONT_00030.mp4"
          "CAMERA_RIGHT_BACK_00080.mp4"
          "CAMERA_RIGHT_BACK_00070.mp4"
          "CAMERA_RIGHT_BACK_00030.mp4"
          "CAMERA_RIGHT_FRONT_00080.mp4"
          "CAMERA_RIGHT_FRONT_00070.mp4"
          "CAMERA_RIGHT_FRONT_00030.mp4")
PROMPT_FILE="/eva_data5/kuoyuhuan/DLP_final/prompt/video_prompts.json"
CKPT_PATH="/eva_data5/kuoyuhuan/DLP_final/DiffusionAsShader/training/v5/checkpoint-1500"
# CKPT_PATH=${CKPT_PATH}/`ls $CKPT_PATH | tail -n 1`  # Get the latest checkpoint

for VID in "${VID_NAME[@]}"; do
    PROMPT=$(cat $PROMPT_FILE | jq -r '.["'$VID_NAME'"]')

    # accelerate launch --config_file accelerate_config.yaml \
    python \
        inference.py \
        --prompt "$PROMPT" \
        --image_or_video_path "/eva_data5/kuoyuhuan/DLP_final/data/cam_video/$VID_NAME" \
        --model_path "THUDM/CogVideoX-2B" \
        --transformer_path "$CKPT_PATH" \
        --output_path "./out_${VID}" \
        --guidance_scale 6.0 \
        --num_inference_steps 50 \
        --generate_type "i2v" \
        --dtype "float16" \
        --seed 1 \
        --num_frames 52 \
        --tracking_path "/eva_data5/kuoyuhuan/DLP_final/data/tracking_map/$VID_NAME"
done
