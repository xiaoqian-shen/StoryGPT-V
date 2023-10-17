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
import open_clip

parser = argparse.ArgumentParser(description='Evaluate Frechet Story and Image distance')
parser.add_argument('--gen_dir', type=str, required=True)
parser.add_argument('--root_path', type=str, default='/ibex/ai/home/shenx/story/data')
parser.add_argument('--dataset', type=str, default='flintstones')
parser.add_argument('--mode', type=str, default='test')
args = parser.parse_args()
args.ref_dir = os.path.join(args.root_path, args.dataset)
print(args.gen_dir)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.load_state_dict(torch.load(os.path.join(args.ref_dir, 'clip_ft.pt')))
model.eval()

class EvalData(Dataset):
    def __init__(self, root_path, gen_path):
        self.root_path = root_path
        with open(os.path.join(self.root_path, 'split.json'),'r') as f:
            self.image_ids = json.load(f)['test']
        with open(os.path.join(self.root_path, 'cleaned_annotations.json'),'r') as f:
            self.ann_file = json.load(f)
        self.gen_path = gen_path
        gen_ids=[filename.split('.')[0] for filename in os.listdir(self.gen_path)]
        self.image_ids = gen_ids
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        video_meta_path = os.path.join(self.root_path, 'video_frames_sampled_4x', f'{image_id}.npy')
        video_array = np.load(video_meta_path)
        n_frames = video_array.shape[0]
        random_range = random.randrange(n_frames)
        image = Image.fromarray(video_array[random_range])
        image_gt = preprocess(image)
        image_path = os.path.join(self.gen_path, f'{image_id}.png')
        image = Image.open(image_path)
        image_gen = preprocess(image)
        try:
            caption = self.ann_file[image_id]['caption']
        except:
            idx = int(np.ceil((random_range + 1) / n_frames * len(self.ann_file[image_id]['captions'])))
            idx = min(idx, len(self.ann_file[image_id]['captions']) - 1)
            caption = self.ann_file[image_id]['captions'][idx]
        return image_gen, image_gt, caption

dataset = EvalData(root_path=args.ref_dir, gen_path=args.gen_dir)
dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)
sims = []
for image_gen, image_gt, captions in tqdm(dataloader):
    with torch.no_grad(), torch.cuda.amp.autocast():
        captions = tokenizer(captions)
        image_gen_features = model.encode_image(image_gen)
        image_gt_features = model.encode_image(image_gt)
        text_features = model.encode_text(captions)
        image_gen_features /= image_gen_features.norm(p=2, dim=-1, keepdim=True)
        image_gt_features /= image_gt_features.norm(p=2, dim=-1, keepdim=True)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        ti_sim_gen = (image_gen_features * text_features).sum(axis=-1).mean(0)
        ti_sim_gt = (image_gt_features * text_features).sum(axis=-1).mean(0)
        ii_sim = (image_gt_features * image_gen_features).sum(axis=-1).mean(0)
        sims.append([ti_sim_gen, ti_sim_gt, ii_sim])
sims = np.array(sims)
print(np.average(sims, axis=0))

