import argparse
from pipelines import InPaintDDIM
import time
import os
from pathlib import Path
import torch
import numpy as np
import PIL
from draw import draw,get_mask
from torchvision import transforms
from image_similarity_measures.evaluate import evaluation

def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    pipe = InPaintDDIM.from_pretrained(args.model)
    size=pipe.unet.config.sample_size
    mask = torch.permute(torch.tensor(get_mask(size,type='R',coverage=0.1), dtype=torch.float32),(2,0,1))

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(".bmp") and "mask" not in file.lower():
                input_path = os.path.join(root, file)
                basename = Path(input_path).stem
                try:
                    image = PIL.Image.open(input_path).convert("RGB").resize((size,size))
                    image_tensor = (transform(image)-0.5)*2
                    pipe.to("cuda:0")
                    image = pipe(ref_image=image_tensor,mask=mask,num_inference_steps=60)
                    predicted_img = f"/tmp/inpainted_{int(time.time())}.png"
                    image.images[0].save(f"{predicted_img}")
                    # print out image similarity:
                    sim = evaluation(org_img_path=f"{input_path}", 
                        pred_img_path=f"{predicted_img}", 
                        metrics=["rmse", "psnr", "fsim", "ssim", "uiq"])
                    print(f"Image Similarity for {basename}: {sim}")

                except Exception as e:
                    print(f"Error masking {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InPaint')
    parser.add_argument('--input_dir', type=str, help='input image dir')
    parser.add_argument('--model', type=str, help='pretrained model')
    args = parser.parse_args()
    main(args)
