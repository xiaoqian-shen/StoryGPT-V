import torch
from lavis.models import load_model, load_model_and_preprocess
from PIL import Image

import torchvision.transforms as transforms
import argparse
import os

import torch
import json
import numpy as np
import random
import os
import tempfile
from PIL import Image

from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

def compute_max(scorer, pred_prompts, gt_prompt):
    scores = []
    for pred_prompt in pred_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
    return np.max(scores)

parser = argparse.ArgumentParser(description='Evaluate Frechet Story and Image distance')
parser.add_argument('--gen_dir', type=str, required=True)
parser.add_argument('--root_path', type=str, default='/ibex/ai/home/shenx/story/data')
parser.add_argument('--dataset', type=str, default='flintstones')
parser.add_argument('--mode', type=str, default='test')

args = parser.parse_args()
args.ref_dir = os.path.join(args.root_path, args.dataset)
print(args.gen_dir)
captioner_ckpt = os.path.join(args.ref_dir, 'captioner.pth')
blip2, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device='cuda'
        )
checkpoint = torch.load(captioner_ckpt, map_location='cpu')
state_dict = checkpoint["model"]
blip2.load_state_dict(state_dict, strict=False)

data_name=args.dataset
data_path=os.path.join(args.root_path, data_name)
annotations = json.load(open(os.path.join(data_path, 'cleaned_annotations.json'), 'r'))

cider_scorer = Cider()
bleu1 = Bleu(n=1)
bleu2 = Bleu(n=2)
bleu3 = Bleu(n=3)
bleu4 = Bleu(n=4)

cider_scores_pred = {}
cider_scores_gt = {}
bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

bleu1_scores_img = []
bleu2_scores_img = []
bleu3_scores_img = []
bleu4_scores_img = []

instance_id = 0

for story_id in tqdm(os.listdir(args.gen_dir)):
    if not str(story_id).endswith('.png'):
        story_path = os.path.join(args.gen_dir, story_id)
        for image_name in os.listdir(story_path):
            image_path = os.path.join(story_path, image_name)
            image_gen = Image.open(image_path).convert("RGB")

            image_id = image_name.split('.')[0]
            try:
                caption = annotations[image_id]['caption']
            except:
                caption = random.sample(annotations[image_id]['captions'], 1)[0]

            video_meta_path = os.path.join(data_path, 'video_frames_sampled_4x', f'{image_id}.npy')
            video_array = np.load(video_meta_path)
            n_frames = video_array.shape[0]
            random_range = random.randrange(n_frames)
            image_gt = Image.fromarray(video_array[random_range]).convert("RGB")

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_gt = vis_processors["eval"](image_gt).unsqueeze(0).to('cuda')
                pred_cap_gts = blip2.generate({"image": image_gt}, num_captions=5)

                image_gen = vis_processors["eval"](image_gen).unsqueeze(0).to('cuda')
                pred_cap_gens = blip2.generate({"image": image_gen}, num_captions=1)
                
                cider_scores_pred[instance_id] = pred_cap_gens
                cider_scores_gt[instance_id] = pred_cap_gts
                instance_id += 1

                pred_cap_gts = [pred_cap_gts[0]]
                bleu1_scores.append(bleu1.compute_score({0: pred_cap_gens}, {0: [caption]})[0])
                bleu2_scores.append(bleu2.compute_score({0: pred_cap_gens}, {0: [caption]})[0])
                bleu3_scores.append(bleu3.compute_score({0: pred_cap_gens}, {0: [caption]})[0])
                bleu4_scores.append(bleu4.compute_score({0: pred_cap_gens}, {0: [caption]})[0])

                bleu1_scores_img.append(bleu1.compute_score({0: pred_cap_gens}, {0: pred_cap_gts})[0])
                bleu2_scores_img.append(bleu2.compute_score({0: pred_cap_gens}, {0: pred_cap_gts})[0])
                bleu3_scores_img.append(bleu3.compute_score({0: pred_cap_gens}, {0: pred_cap_gts})[0])
                bleu4_scores_img.append(bleu4.compute_score({0: pred_cap_gens}, {0: pred_cap_gts})[0])

print('compare to gt caption: ', np.mean(bleu1_scores),np.mean(bleu2_scores),np.mean(bleu3_scores),np.mean(bleu4_scores), flush=True)
print('compare to gt image: ', np.mean(bleu1_scores_img),np.mean(bleu2_scores_img),np.mean(bleu3_scores_img),np.mean(bleu4_scores_img), flush=True)
score, scores = cider_scorer.compute_score(cider_scores_gt, cider_scores_pred)
print(score, np.max(scores))
