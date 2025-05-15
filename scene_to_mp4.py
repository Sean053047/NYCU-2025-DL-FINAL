#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export sweep‑only front‑camera frames of one TruckScenes *scene* into multiple
MP4 videos.  Each video contains **exactly N (default 81) consecutive sweep
frames**; any剩餘不足 N 幀的尾段會被捨棄。

依賴套件
--------
    pip install truckscenes-devkit opencv-python tqdm

使用範例
--------
```bash
python scene_to_sweep_videos.py \
  --dataroot /data/man-truckscenes \
  --version v1.0-trainval \
  --scene 12 \
  --channel CAMERA_LEFT_FRONT \
  --chunk-size 81 \
  --out ./videos
```
會產生：
```
videos/
 ├─ scene-012_CAMERA_LEFT_FRONT_part0.mp4  (81 frames)
 ├─ scene-012_CAMERA_LEFT_FRONT_part1.mp4  (81 frames)
 └─ ...
```
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm
from truckscenes import TruckScenes
import hashlib
import numpy as np

name_idx = 0

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def collect_sweep_frame_records(ts: TruckScenes,
                         scene_token: str,
                         camera_channel: str):
    """一次走完整條 sample_data → next 鏈，只保留 sweeps 影格且不重複。"""
    scene = ts.get('scene', scene_token)
    first_sample = ts.get('sample', scene['first_sample_token'])
    sd_token = first_sample['data'][camera_channel]

    frames, seen_tokens = [], set()
    while sd_token:
        if sd_token in seen_tokens:          # 理論上不會發生，保險
            break
        seen_tokens.add(sd_token)

        sd = ts.get('sample_data', sd_token)
        if not sd['is_key_frame']:           # 只要 sweeps
            fpath = os.path.join(ts.dataroot, sd['filename'])
            frames.append((fpath, sd['timestamp']))
        sd_token = sd['next']
    return frames

def write_video(frames, out_path, fps):
    """給定 (fpath, ts) list 寫 mp4；自動過濾同檔名或同 md5 的重複影格。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # first_img = cv2.imread(frames[0][0])
    # h, w = first_img.shape[:2]
    w, h = 832, 480 
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    seen_hash = set()
    for fpath, _ in frames:
        buf = open(fpath, 'rb').read()
        md5 = hashlib.md5(buf).digest()
        if md5 in seen_hash:                 # 完全相同畫面就跳過
            continue
        seen_hash.add(md5)
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (w, h))      # resize to 832x480
        vw.write(img)
    vw.release()


def process_scene(ts, scene_index, channel, chunk_size, out_dir):
    """Process a scene by its index and export sweep-only videos."""
    scene_token = (ts.scene[int(scene_index)]["token"]
                    if str(scene_index).isdigit() else scene_index)

    # 取得 sweep frame 列表
    frames = collect_sweep_frame_records(ts, scene_token, channel)
    if len(frames) < chunk_size:
        raise RuntimeError("此 scene 的 sweeps 張數不足一段影片。")

    # fps = guess_fps(frames)
    fps = 10
    scene_name = ts.get("scene", scene_token)["name"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 分段寫入
    num_chunks = len(frames) // chunk_size  # discard remainder
    for idx in range(num_chunks + 1):
        start = idx * chunk_size
        end = start + chunk_size
        chunk_frames = []
        if idx != num_chunks:
            chunk_frames = frames[start:end]
        else:
            chunk_frames = frames[len(frames) - chunk_size:]
        out_path = out_dir / f"{scene_index}_{channel}_{idx}.mp4"
        write_video(chunk_frames, out_path, fps)

# -----------------------------------------------------------
# Main script
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export sweep‑only videos (chunked) for one TruckScenes scene")
    parser.add_argument("--dataroot", required=True,
                        help="資料根目錄，例如 /data/man-truckscenes")
    parser.add_argument("--version", default="v1.0-mini",
                        help="資料分割版本 (trainval / mini / test)")
    parser.add_argument("--chunk-size", type=int, default=81,
                        help="每部影片的幀數，預設 81")
    parser.add_argument("--out", default=".", help="輸出資料夾")
    args = parser.parse_args()

    ts = TruckScenes(args.version, args.dataroot, verbose=True)
    
    channels = ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT", "CAMERA_LEFT_BACK", "CAMERA_RIGHT_BACK"]
    
    for channel in channels:
        for scene in tqdm(range(len(ts.scene))):
            process_scene(ts, scene, channel, args.chunk_size, args.out)


if __name__ == "__main__":
    main()
