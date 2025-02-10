from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64  # assumes images are square
    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 300
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "HUST-64-300-finetuned"  # the fine-turned model name
    pretrained_model = "HUST-64-300/epoch299"
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    dataset_name="/home/ts/Downloads/obs/HUST-OBC/deciphered_64x64"

config = TrainingConfig()

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
import os
from accelerate import notebook_launcher
import numpy as np
import random
import matplotlib.pyplot as plt
from pipelines import InPaintDDIM

def normalize_neg_one_to_one(img):
    return img * 2 - 1

class LocalDataset(Dataset):
    # A dataset that loads images from a folder
    def __init__(self, folder, image_size, exts=['bmp','png','jpg','jpeg']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(folder).glob(f'**/*.{ext}')]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(normalize_neg_one_to_one),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('L')  # Open the image in grayscale mode
        return self.transform(img)

# Fine-tuning pretrained model using inpainting mask loss
dataset = LocalDataset(config.dataset_name, image_size=config.image_size)
train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

device = torch.device("cpu") #("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Using device: cuda

# Load the pretrained model to be fine-tuned
pipe = DDPMPipeline.from_pretrained('/home/ts/ddimtrain/HUST-64-Aug300/epoch299').to(device)
model = pipe.unet

optimizer = AdamW(model.parameters(), lr=config.learning_rate)
inference_scheduler = DPMSolverMultistepScheduler()
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    global_step = 0
    model.train()
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=False)
        progress_bar.set_description(f"Epoch {epoch}")
        for images in train_dataloader:
            images = images.to(device)
            predicted_imgs = pipe(
                batch_size=32, 
                generator=torch.Generator(device=device).manual_seed(config.seed), # Generator can be on GPU here
                num_inference_steps=50,
            ).images
            predicted_array = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in predicted_imgs])  # (BatchSize, Height, Width,)
            predicted_array = np.expand_dims(predicted_array, axis=-1)             # (BatchSize, Height, Width, Channel)
            predicted_tensor = torch.tensor(predicted_array, dtype=torch.uint8)    # (BatchSize, Height, Width, Channel)
            predicted_tensor = predicted_tensor.permute(0, 3, 1, 2).to(device)     # (BatchSize, Channel, Height, Width)

            # BP
            optimizer.zero_grad()
            loss = F.mse_loss(predicted_tensor, images).requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            print("parameter device:", next(model.parameters()).device)  # parameter device: cuda:0

args = (config, model, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)