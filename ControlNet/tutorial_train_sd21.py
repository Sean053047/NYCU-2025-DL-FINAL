from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

# Configs
resume_path = './lightning_logs/version_16/checkpoints/epoch=0-step=999.ckpt'
batch_size = 1
logger_freq = 100
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

def train(model):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    


    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=3, 
        strategy="ddp",
        # min_epochs=10,
        precision=32,
        max_steps = 5000, 
        callbacks=[logger]
    )

    # Train!
    trainer.fit(model, dataloader)
    
def validate(model):
    model.eval()
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
    for batch in dataloader:
        with torch.no_grad():
            model.set_input(batch)
            model.test()
            model.log_images(batch, step=0, prefix='val')
            print(f"Processed batch with prompt: {batch['txt'][0]}")
        break
    
if __name__ == '__main__':
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    train(model)
    # validate(model)
