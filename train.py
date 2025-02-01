
from tqdm.auto import tqdm
from pathlib import Path
import os
from accelerate import notebook_launcher
from huggingface_hub import notebook_login
from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DDIMScheduler
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import math
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami


def load_image_dataset(data_dir):
  image_files = []
  for file_path in Path(data_dir).rglob("*"):
    if file_path.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']:
      image_files.append(str(file_path))

  dataset_dict = {'train': [{'file_path': path} for path in image_files]}
  print(dataset_dict)
  dataset = load_dataset('imagefolder', data_dir=data_dir, data_files=dataset_dict)
  return dataset

@dataclass
class TrainingConfig:
    image_size = 64
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = 'fp16'
    output_dir = 'obs-handprint-64LR1E5bz16EP100-DDIM'
    save_result_epochs = 10
    overwrite_output_dir = True
    seed = 12345

config = TrainingConfig()
data_dir = "/home/ts/Downloads/obs/handprint_64x64"
dataset = load_dataset("imagefolder", data_dir=data_dir, split="train") 
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(imgs):
    images = [preprocess(image.convert("RGB")) for image in imgs["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 64, 128, 128, 256, 256),
    down_block_types=( 
        "DownBlock2D",
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
)

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
sample_image = dataset[0]['images'].unsqueeze(0)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
# noisy_image.permute(0, 2, 3, 1): This part deals with the tensor's dimensions. 
# PyTorch often uses a channel-first format (NCHW - Batch, Channels, Height, Width). 
# PIL Images, and many image processing libraries, expect channel-last (NHWC - Batch, Height, Width, Channels). 
# permute(0, 2, 3, 1) rearranges the dimensions to move the channels from the second position to the last.
# The 0 remains for the batch dimension, and the other dimensions are reordered.
Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images
    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "proj")
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            batchsize = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batchsize,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            print(f"epoch,{epoch},step,{global_step},loss,{loss.detach().item()},lr,{lr_scheduler.get_last_lr()[0]}")
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_result_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
                pipeline.save_pretrained(config.output_dir) 
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)
