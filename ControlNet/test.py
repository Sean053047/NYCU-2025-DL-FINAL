from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import torch
from torchvision.utils import save_image
import os
import random
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
import einops
import numpy as np
from PIL import Image


# Configs
resume_path = './lightning_logs/version_16/checkpoints/epoch=0-step=999.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

IMAGE_RESOLUTION = 512  

def test(model):
    model.eval()
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
    
    save_dir = './test_results'
    os.makedirs(save_dir, exist_ok=True)    
    ddim_sampler = DDIMSampler(model)

    # Inference param
    num_samples = 1
    ddim_steps = 100
    guess_mode = False
    strength = 1.5
    scale = 3.0
    eta = 0.0
    seed = -1
    
    import decord
    vid = [
        "CAMERA_LEFT_BACK_00030.mp4",
        "CAMERA_LEFT_BACK_00070.mp4",
        "CAMERA_LEFT_BACK_00080.mp4",
        "CAMERA_LEFT_FRONT_00030.mp4",
        "CAMERA_LEFT_FRONT_00070.mp4",
        "CAMERA_LEFT_FRONT_00080.mp4",
        "CAMERA_RIGHT_BACK_00030.mp4",
        "CAMERA_RIGHT_BACK_00070.mp4",
        "CAMERA_RIGHT_BACK_00080.mp4",
        "CAMERA_RIGHT_FRONT_00030.mp4"
        "CAMERA_RIGHT_FRONT_00070.mp4",
        "CAMERA_RIGHT_FRONT_00080.mp4",
    ]
    idxs = [
        29,
        69,
        79,
        29+1045,
        69+1045,
        79+1045,
        29+1045+588,
        69+1045+588,
        79+1045+588,
        29+1045+588+588,
        69+1045+588+588,
        79+1045+588+588
    ]
    
    batch_idx = 0
    for v, idx in zip(vid, idxs):
        idx = idx * 81
        # idx = random.randint(0, len(dataset) - 1)
        batch = dataset[idx]
        with torch.no_grad():
            con = batch['hint']
            raw_gt = batch['jpg']
            prompt = batch['txt']

            # Save GT
            if isinstance(raw_gt, torch.Tensor):
                gt_np = raw_gt.cpu().numpy()
            else:
                gt_np = raw_gt
            if gt_np.dtype != np.uint8:
                gt_np = (gt_np * 255.0).clip(0,255).astype(np.uint8)
            gt_rgb = HWC3(gt_np)
            gt_img = Image.fromarray(gt_rgb)
            # gt_path = os.path.join(save_dir, f'gt_{batch_idx:04d}.png')
            gt_path = os.path.join(save_dir, f'gt_{v}.png')
            gt_img.save(gt_path)
            print(f"Saved ground truth image: {gt_path}")

            if isinstance(con, torch.Tensor):
                con_np = con.cpu().numpy()
            else:
                con_np = con 

            if con_np.ndim == 4:
                con_np = con_np[0] 
            con_np = (con_np * 255.0).clip(0,255).astype(np.uint8)
            con_rgb = HWC3(con_np)  # → shape=(H0, W0, 3)、dtype=uint8
                    
            con_tensor = torch.from_numpy(con_rgb.copy()).float().cuda() / 255.0
            con_tensor = torch.stack([con_tensor for _ in range(num_samples)], dim=0)  # → (1, H, W, 3)
            con_tensor = einops.rearrange(con_tensor, 'b h w c -> b c h w').clone()

            _, C, H, W = con_tensor.shape
            
            if seed == -1:
                current_seed = random.randint(0, 65535)
            else:
                current_seed = seed
            seed_everything(current_seed)

            a_prompt = 'best quality, extremely detailed'
            n_prompt = 'longbody, lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality'
            
            cond = {
                "c_concat": [con_tensor], 
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
            }
            un_cond = {
                "c_concat": None if guess_mode else [con_tensor], 
                "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
            }
            
            shape = (4, H // 8, W // 8)
            
            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            # DDIM
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, num_samples,
                shape, cond, verbose=False, eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond
            )
            
            # Decode
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            
            for i in range(num_samples):
                gen_img = Image.fromarray(x_samples[i])
                # gen_path = os.path.join(save_dir, f'gen_{batch_idx:04d}_{i:02d}.png')
                gen_path = os.path.join(save_dir, f'gen_{v}.png')
                gen_img.save(gen_path)
                
                con_np = con_tensor[0].cpu().numpy()  # shape = [3, H, W]
                con_np = np.transpose(con_np, (1, 2, 0))  # → [H, W, 3]
                con_np = (con_np * 255).astype(np.uint8)
                con_img = Image.fromarray(con_np)
                # con_path = os.path.join(save_dir, f'con_{batch_idx:04d}_{i:02d}.png')
                con_path = os.path.join(save_dir, f'con_{v}.png')
                con_img.save(con_path)
                
                print(f"Saved generated image: {gen_path}")
                print(f"Saved con image: {con_path}")
                
            combined_img = Image.new('RGB', (gen_img.width + gt_img.width, gen_img.height))
            combined_img.paste(gt_img, (0, 0))
            combined_img.paste(gen_img, (gt_img.width, 0))
            combined_img_path = os.path.join(save_dir, f'combined_{v}.png')
            combined_img.save(combined_img_path)
            print(f"Saved combined image: {combined_img_path}")

            print(f"Used seed: {current_seed}")

if __name__ == '__main__':
    # Clear GPU cache first
    torch.cuda.empty_cache()
    
    # Load model on CPU first
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    
    gpu = 'cuda:0'
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()  # Clear cache for this specific GPU
    model = model.to(gpu)
    device = torch.device(gpu)
    print(f"Successfully moved model to {device}")
    
    test(model)