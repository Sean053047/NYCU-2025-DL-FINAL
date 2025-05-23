import os
import glob
import random

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from controlnet_aux import CannyDetector, HEDdetector
from safetensors.torch import load_file, save_file
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

def unpack_mm_params(p):
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


def resize_for_crop(image, min_h, min_w):
    img_h, img_w = image.shape[-2:]
    
    if img_h >= min_h and img_w >= min_w:
        coef = min(min_h / img_h, min_w / img_w)
    elif img_h <= min_h and img_w <=min_w:
        coef = max(min_h / img_h, min_w / img_w)
    else:
        coef = min_h / img_h if min_h > img_h else min_w / img_w 

    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def init_controlnet(controlnet_type):
    if controlnet_type in ['canny']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
}


class BaseClass(Dataset):
    def __init__(
            self, 
            video_root_dir,
            image_size=(320, 512), 
            stride=(1, 2), 
            sample_n_frames=25,
            hflip_p=0.5,
            controlnet_type='canny',
        ):
        self.height, self.width = unpack_mm_params(image_size)
        self.stride_min, self.stride_max = unpack_mm_params(stride)
        self.video_root_dir = video_root_dir
        self.sample_n_frames = sample_n_frames
        self.hflip_p = hflip_p
        
        self.length = 0
        
        # self.controlnet_processor = init_controlnet(controlnet_type)
        
    def __len__(self):
        return self.length
        
    def load_video_info(self, video_path):
        video_reader = VideoReader(video_path)
        fps_original = video_reader.get_avg_fps()
        video_length = len(video_reader)
        
        sample_stride = random.randint(self.stride_min, self.stride_max)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        np_video = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 127.5 - 1
        del video_reader
        controlnet_video = [self.controlnet_processor(x) for x in np_video]
        controlnet_video = torch.from_numpy(np.stack(controlnet_video)).permute(0, 3, 1, 2).contiguous()
        controlnet_video = controlnet_video / 127.5 - 1
        return pixel_values, controlnet_video
        
    def get_batch(self, idx):
        raise Exception('Get batch method is not realized.')

    def __getitem__(self, idx):
        while True:
            try:
                video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        if self.hflip_p > random.random():
            video, controlnet_video = [
                transforms.functional.hflip(x) for x in [video, controlnet_video]
            ]
            
        video, controlnet_video = [
            resize_for_crop(x, self.height, self.width) for x in [video, controlnet_video]
        ] 
        video, controlnet_video = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [video, controlnet_video]
        ]
        data = {
            'video': video, 
            'caption': caption, 
            'controlnet_video': controlnet_video,
        }
        return data
    

def encode_video(accelerator, vae, video):
    video = video.to(accelerator.device, dtype=vae.dtype)
    video = video.view(-1, video.shape[0], video.shape[1], video.shape[2], video.shape[3])
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds

class TrucksceneControlnetDataset(Dataset):
    def __init__(self, 
            video_root_dir,
            csv_path,
            image_size,
            stride,
            sample_n_frames,
            vae,
            tokenizer,
            text_encoder,
            accelerator,
            model_config,
            weight_dtype,
        ):
        super().__init__()
        self.video_root_dir = video_root_dir
        self.df = pd.read_csv(csv_path)
        self.width, self.height = unpack_mm_params(image_size)
        self.stride_min, self.stride_max = unpack_mm_params(stride)
        self.sample_n_frames = sample_n_frames
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.accelerator = accelerator
        self.model_config = model_config
        self.weight_dtype = weight_dtype
        self.length = self.df.shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        pixel_values, caption, controlnet_video = self.get_batch(idx)

        data = {
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_video,
        }
        return data
    
    def get_batch(self, idx):
        item = self.df.iloc[idx]
        caption = item['prompt']
        video_name = item['name'].replace('.mp4', '.pt')
        
        pixel_path = os.path.join(self.video_root_dir, 'cam_video', video_name)
        # controlnet_path = os.path.join(self.video_root_dir, 'lidar_video', video_name)
        controlnet_path = pixel_path
        caption_path = os.path.join(self.video_root_dir, 'captions', video_name)
        
        if os.path.exists(pixel_path) and os.path.exists(controlnet_path) and os.path.exists(caption_path):
            pixel_values = load_file(pixel_path)['pixel_values']
            controlnet_video = load_file(controlnet_path)['pixel_values']
            # controlnet_video = load_file(controlnet_path)['controlnet_video']
            caption = load_file(caption_path)['caption']
            # _, controlnet_video = self.load_video_info(video_name.replace('.pt', '.mp4'))
        else:
            pixel_values, controlnet_video = self.load_video_info(video_name.replace('.pt', '.mp4'))
            pixel_emb = encode_video(self.accelerator, self.vae, pixel_values).contiguous()
            controlnet_emb = encode_video(self.accelerator, self.vae, controlnet_video).contiguous()
            caption_emb = compute_prompt_embeddings(
                    self.tokenizer,
                    self.text_encoder,
                    caption,
                    self.model_config.max_text_seq_length,
                    self.accelerator.device,
                    self.weight_dtype,
                    requires_grad=False,
                ).contiguous()

            save_file({'pixel_values': pixel_emb}, pixel_path)
            # save_file({'controlnet_video': controlnet_emb}, controlnet_path)
            save_file({'caption': caption_emb}, caption_path)

            pixel_values = pixel_emb
            controlnet_video = controlnet_emb
            caption = caption_emb     

        return pixel_values[0], caption[0], controlnet_video[0]
    
    def load_video_info(self, video_name):
        video_reader = VideoReader(os.path.join(self.video_root_dir, 'cam_video', video_name))
        fps_original = video_reader.get_avg_fps()
        video_length = len(video_reader)
        
        sample_stride = random.randint(self.stride_min, self.stride_max)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        np_video = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 127.5 - 1
        del video_reader
        
        # video_reader = VideoReader(os.path.join(self.video_root_dir, 'lidar_video', video_name))
        # fps_original = video_reader.get_avg_fps()
        # video_length = len(video_reader)
        
        # sample_stride = random.randint(self.stride_min, self.stride_max)
        # clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        # start_idx   = random.randint(0, video_length - clip_length)
        # batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        # np_video = video_reader.get_batch(batch_index).asnumpy()
        # pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous()
        # pixel_values = pixel_values / 127.5 - 1
        # del video_reader

        
        # return pixel_values, controlnet_video
        return pixel_values, pixel_values.clone()
