set -e -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
VID_NAME="CAMERA_LEFT_BACK_00001.mp4"
PROMPT_FILE="/eva_data5/kuoyuhuan/DLP_final/prompt/video_prompts.json"
PROMPT=$(cat $PROMPT_FILE | jq -r '.["'$VID_NAME'"]')

# accelerate launch --config_file accelerate_config.yaml \
python \
    inference.py \
    --prompt "$PROMPT" \
    --image_or_video_path "/eva_data5/kuoyuhuan/DLP_final/dl_final_test/cam_video/$VID_NAME" \
    --model_path "THUDM/CogVideoX-2B" \
    --transformer_path "/eva_data5/kuoyuhuan/DLP_final/DiffusionAsShader/training/v2/checkpoint-1500" \
    --output_path "./output.mp4" \
    --guidance_scale 6.0 \
    --num_inference_steps 25 \
    --generate_type "i2v" \
    --dtype "float16" \
    --seed 42 \
    --tracking_path "/eva_data5/kuoyuhuan/DLP_final/dl_final_test/lidar_video/$VID_NAME"
