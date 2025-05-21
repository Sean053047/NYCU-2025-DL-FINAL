import os
import json
import argparse
from tqdm import tqdm

import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor


def extract_frames(video_path, num_frames):
    """從影片中抽取代表性 frames（均勻取樣）"""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    return np.stack(frames)


def load_videollava(model_id="LanguageBind/Video-LLaVA-7B-hf"):
    print(f"Loading Video-LLaVA model: {model_id}")
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    processor = VideoLlavaProcessor.from_pretrained(model_id)
    return model, processor


def generate_prompt(model, processor, video_path, prompt_text, num_frames, max_new_tokens):
    frames = extract_frames(video_path, num_frames)
    inputs = processor(videos=[frames], text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device, dtype=torch.float16 if v.dtype == torch.float32 else None)
              for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens= max_new_tokens)

    decoded = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return decoded.split("ASSISTANT:")[-1].strip()      # 去掉原始 prompt


def main(args):
    model, processor = load_videollava()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])

    for fname in tqdm(video_files, desc="Generating prompts"):
        fpath = os.path.join(args.video_dir, fname)
        try:
            caption = generate_prompt(model, processor, fpath, args.prompt_for_VLM, args.num_frames, args.max_new_tokens)
            results[fname] = caption
        except Exception as e:
            print(f"Error on {fname}: {e}")
            results[fname] = "[ERROR]"

    out_path = os.path.join(args.output_dir, "video_prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved prompts to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text prompts for videos")
    parser.add_argument("--num-frames", type=int, default=10, help="get key frames from video")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="maximum tokens to generate")
    # parser.add_argument("--prompt-for-VLM", type=str, default="USER: <video>\nDescribe what is visually observable in the video frame by frame. Do **not** include any subjective interpretation or inferred actions. Do **not** include any irrelevant information such as \"In the video...\", \"The video shows...\". Include the elements such as perspective direction of the camera, nearby vehicles, lane markings, road structure, environment, and weather. ASSISTANT:", help="prompt for VLM to describe the video")
    parser.add_argument("--prompt-for-VLM", type=str, default="USER: <video>\nDescribe the video in detail, specifying the direction in which the car is traveling (e.g. moving forward, turning left), emphasizing that it is captured from a first-person, in-car perspective, and listing all other objects and elements in the scene—such as surrounding vehicles, pedestrians, traffic signals, road signs, buildings, and environmental features. ASSISTANT:", help="prompt for VLM to describe the video")
    parser.add_argument("--video-dir", type=str, default="./videos", help="input video directory path")
    parser.add_argument("--output-dir", type=str, default="./prompt", help="output .json directory path")

    args = parser.parse_args()
    main(args)
