import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import argparse
import os

import torch
import json
import numpy as np
import random
import os
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EPOCH = 40

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = open_clip.create_model_from_pretrained('ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

class ImageCaptionDataset(Dataset):
    def __init__(self, root_path, split='train'):
        self.root_path = root_path
        with open(os.path.join(self.root_path, 'split.json'),'r') as f:
            self.image_ids = json.load(f)[split]
        with open(os.path.join(self.root_path, 'cleaned_annotations.json'),'r') as f:
            self.ann_file = json.load(f)
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        video_meta_path = os.path.join(self.root_path, 'video_frames_sampled_4x', f'{image_id}.npy')
        video_array = np.load(video_meta_path)
        n_frames = video_array.shape[0]
        random_range = random.randrange(n_frames)
        image = Image.fromarray(video_array[random_range])
        image = preprocess(image).half()
        try:
            caption = self.ann_file[image_id]['caption']
        except:
            idx = int(np.ceil((random_range + 1) / n_frames * len(self.ann_file[image_id]['captions'])))
            idx = min(idx, len(self.ann_file[image_id]['captions']) - 1)
            caption = self.ann_file[image_id]['captions'][idx]
        # caption = tokenizer(caption)
        return image, caption

root_path = '/ibex/ai/home/shenx/story/data/pororo'
train_dataset = ImageCaptionDataset(root_path)
train_dataloader = DataLoader(train_dataset, batch_size = 256)
eval_dataset = ImageCaptionDataset(root_path)
eval_dataloader = DataLoader(eval_dataset, batch_size = 512)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

open_clip.model.convert_weights_to_lp(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)

best_sim = 0.0

for epoch in tqdm(range(EPOCH)):
    total_loss = []
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(True):
            images,texts = batch 
            texts = tokenizer(texts)
            images= images.to(device)
            texts = texts.to(device)

            image_features, text_features, logit_scale = model(images, texts)

            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            loss.backward()
            total_loss.append(loss.item())
        convert_models_to_fp32(model)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        optimizer.step()
        open_clip.model.convert_weights_to_lp(model)
    print(f'[Epoch {epoch}] Loss: ', np.average(total_loss), flush=True)

    model.eval()
    sims = []
    for batch in eval_dataloader:
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            images,texts = batch
            texts = tokenizer(texts)
            images= images.to(device)
            texts = texts.to(device)
            image_features, text_features, logit_scale = model(images, texts)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
            sim = (image_features * text_features).sum(axis=-1).mean(0).cpu()
            sims.append(sim)
    print(f'[Epoch {epoch}] CLIP Score: ', np.average(sims), flush=True)

    if np.average(sims) > best_sim:
        best_sim = np.average(sims)
        torch.save(model.state_dict(), "clip_best.pt")
