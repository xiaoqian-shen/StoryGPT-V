import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json
import pickle
import eval_cls
import random
import argparse
from fid_score import fid_score
from tqdm import tqdm

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

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--dataset', type=str, default="flintstones")
    parser.add_argument('--root_path', type=str, default="/ibex/ai/home/shenx/story/data")
    parser.add_argument('--gen_dir', type=str)
    return parser.parse_args()

args = parse_args()

data_name=args.dataset
data_path=os.path.join(args.root_path, data_name)

char_label = pickle.load(open(os.path.join(data_path, 'labels.pkl'), 'rb'))
if data_name == 'flintstones':
    eval_char_cls, input_size = eval_cls.initialize_model(model_name='inception', num_classes=7)
    eval_bg_cls, input_size = eval_cls.initialize_model(model_name='inception', num_classes=323)
    eval_char_cls = eval_char_cls.cuda()
    eval_bg_cls = eval_bg_cls.cuda()
    eval_char_cls.load_state_dict(torch.load(os.path.join(data_path, 'classifier_char.pt')))
    eval_bg_cls.load_state_dict(torch.load(os.path.join(data_path, 'classifier_bg.pt')))
    bg_label = pickle.load(open(os.path.join(data_path, 'labels_bg.pkl'), 'rb'))['label']
else:
    eval_char_cls, input_size = eval_cls.initialize_model(model_name='inception', num_classes=9)
    eval_char_cls.load_state_dict(torch.load(os.path.join(data_path, 'classifier_char.pt')))
    eval_char_cls = eval_char_cls.cuda()

transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

eval_images = []
gt_char_labels = []
gt_bg_labels = []
gt_images = []

for story_id in os.listdir(args.gen_dir):
    if not str(story_id).endswith('.png'):
        story_path = os.path.join(args.gen_dir, story_id)
        gen_images = []
        gt_image =[]
        char_labels = []
        bg_labels = []
        for image_name in os.listdir(story_path):
            image_path = os.path.join(story_path, image_name)
            gen_image = Image.open(image_path)
            gen_images.append(gen_image)
            image_id = image_name.split('.')[0]

            video_meta_path = os.path.join(data_path, 'video_frames_sampled_4x', f'{image_id}.npy')
            video_array = np.load(video_meta_path)
            n_frames = video_array.shape[0]
            random_range = random.randrange(n_frames)
            image_gt = Image.fromarray(video_array[random_range])
            gt_image.append(image_gt)

            char_labels.append(torch.tensor(char_label[image_id]))
            if data_name == 'flintstones':
                bg_labels.append(torch.tensor(bg_label[image_id]))
            
        eval_images.append(torch.stack([transform(gen_image) for gen_image in gen_images]))
        gt_images.append(torch.stack([transform(gt) for gt in gt_image]))

        gt_char_labels.append(torch.stack(char_labels))
        if data_name == 'flintstones':
            gt_bg_labels.append(torch.stack(bg_labels))

fid = fid_score(gt_images, eval_images, cuda=True, normalize=True, batch_size=16)

print('FID: ', fid, flush=True)

acc, f1 = eval_cls.eval_model(eval_char_cls, zip(eval_images, gt_char_labels), 'cuda')
print(f'Character Acc: ', acc, flush=True)
print(f'Character F1: ', f1, flush=True)
if data_name == 'flintstones':
    acc, f1 = eval_cls.eval_model(eval_bg_cls, zip(eval_images, gt_bg_labels), 'cuda')
    print(f'Background Acc: ', acc, flush=True)
    print(f'Background F1: ', f1, flush=True)