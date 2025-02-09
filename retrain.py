# How to run:
# conda activate retrain1
# python retrain.py

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
    save_image_epochs = 1
    save_model_epochs = 1
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

class ImageMaskDataset(Dataset):
    def __init__(self, image_folder, transform=None, mask_size=(24, 24), mask_prob=1):
        self.image_folder = image_folder
        exts=['bmp','png','jpg','jpeg']
        self.image_filenames = [p for ext in exts for p in Path(image_folder).glob(f'**/*.{ext}')]
        self.transform = transform
        self.mask_size = mask_size
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert('L')
        
        # Generate a random binary mask
        mask = np.zeros(image.size, dtype=np.uint8)
        if random.random() < self.mask_prob:
            mask_height = random.randint(16, self.mask_size[1])  # Random height for the mask
            mask_width = random.randint(16, self.mask_size[0])   # Random width for the mask
            
            # Randomly choose the position to place the mask
            top = random.randint(0, image.size[1] - mask_height)
            left = random.randint(0, image.size[0] - mask_width)
            
            mask[top:top + mask_height, left:left + mask_width] = 1  # Set the mask area to 1, unmaksed area leave as is (0)
        mask = Image.fromarray(mask * 255)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
        return image, mask

def make_grid(images, rows, cols):
    # Helper function for making a grid of images
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample from the model and save the images in a grid
    images = pipeline(
        batch_size=config.eval_batch_size, 
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Generator must be on CPU for sampling during training
        num_inference_steps=50,
    ).images

    # Make a grid out of the inverted images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

# Fine-tuning pretrained model using inpainting mask loss
dataset = ImageMaskDataset(config.dataset_name)
train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
# Load the pretrained model to be fine-tuned
pipe = InPaintDDIM.from_pretrained('/home/ts/ddimtrain/HUST-64-Aug300/epoch299')
# Open the model in training mode.
pipe.unet.train()
pipe.to("cuda:0")


# Inpainting mask loss function, returns the MSE of the masked areas of the predicted and original images
def mask_loss(predicted, target, mask):
    """
    Custom loss function with masking.
    predicted: The predicted image tensor.
    target: The ground truth image tensor.
    mask: The binary mask tensor. 1:masked, 0:original
    """
    # Calculate Mean Squared Error (MSE) only where the mask is 1
    return F.mse_loss(predicted * mask, target * mask).requires_grad_(True).to('cuda:0')   #, reduction='mean')

optimizer = AdamW(pipe.unet.parameters(), lr=config.learning_rate)
inference_scheduler = DPMSolverMultistepScheduler()
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything for accelerator
    # model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, lr_scheduler
    # )

    # when the line above is included:  accelerator.prepare() 
    # Traceback (most recent call last):██████████████████████████████████████████████████████████████████████████████▎     | 57/60 [00:00<00:00, 70.32it/s]
    #   File "/home/ts/inpaint0/InPainting/retrain.py", line 211, in <module>
    #     notebook_launcher(train_loop, args, num_processes=1)
    #   File "/home/ts/miniconda3/envs/retrain/lib/python3.12/site-packages/accelerate/launchers.py", line 266, in notebook_launcher
    #     function(*args)
    #   File "/home/ts/inpaint0/InPainting/retrain.py", line 187, in train_loop
    #     optimizer.step()
    #   File "/home/ts/miniconda3/envs/retrain/lib/python3.12/site-packages/accelerate/optimizer.py", line 165, in step
    #     self.scaler.step(self.optimizer, closure)
    #   File "/home/ts/miniconda3/envs/retrain/lib/python3.12/site-packages/torch/cuda/amp/grad_scaler.py", line 448, in step
    #     assert (
    # AssertionError: No inf checks were recorded for this optimizer.

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for batch in train_dataloader:
            images,masks = batch
            images.to('cuda:0')  
            masks.to('cuda:0')  
            
            predicted_imgs = pipe(ref_image=images,mask=masks,num_inference_steps=60).images
            image_array = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in predicted_imgs])  # (BatchSize, Height, Width,)
            image_array = np.expand_dims(image_array, axis=-1)  # (BatchSize, Height, Width, Channel)

            image_tensor = torch.tensor(image_array, dtype=torch.float32)    # (BatchSize, Height, Width, Channel)
            image_tensor = image_tensor.permute(0, 3, 1, 2)                  # (BatchSize, Channel, Height, Width)
            with accelerator.accumulate(model):
                loss = mask_loss(image_tensor, images, masks)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(pipe.unet), scheduler=inference_scheduler)
                evaluate(config, epoch, pipeline)
                print(f'evaluate completed {epoch}')

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(pipe.unet), scheduler=inference_scheduler)
                save_dir = os.path.join(config.output_dir, f"epoch{epoch}")
                pipeline.save_pretrained(save_dir)
                print(f'evaluate completed {epoch} {save_dir}')


args = (config, pipe.unet, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)