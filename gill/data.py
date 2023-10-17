"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple, List

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset

from gill import utils
import pickle
import json
import copy
import random


def collate_fn(batch):
    # batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class StoryDataset(Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        self.root = args.dataset_dir

        self.feature_extractor = utils.get_feature_extractor_for_model(
            args.visual_model, image_size=args.image_size, train=False)
        self.image_size = args.image_size

        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.precision = args.precision
        self.retrieval_token_idx = args.retrieval_token_idx
        self.gen_token_idx = args.gen_token_idx
        self.num_tokens = args.num_tokens
        self.num_clip_tokens = args.num_clip_tokens

        with open(os.path.join(self.root, 'split.json'), 'r') as f:
            self.splits = json.load(f)

        image_ids = self.splits[split]

        self.followings = pickle.load(open(os.path.join(self.root, 'following_cache3.pkl'), 'rb'))
        self.labels = pickle.load(open(os.path.join(self.root, 'labels.pkl'), 'rb'))

        self.characters = {}
        annotations = json.load(open(os.path.join(self.root, 'flintstones_annotations_v1-0.json'), 'r'))
        for sample in annotations:
            self.characters[sample["globalID"]] = [c["entityLabel"].strip() for c in sample["characters"]]

        filtered_followings = {}
        for i, f in self.followings.items():
            if len(f) == 3:
                filtered_followings[i] = f
            else:
                continue
        self.followings = filtered_followings

        self.image_ids = [tid for tid in image_ids if tid in self.followings]
        self.annotations = json.load(open(os.path.join(self.root, 'cleaned_annotations.json'), 'r'))
        self.clip_embs = pickle.load(open(os.path.join(self.root, args.clip_emb_file), 'rb'))

        self.font = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        female = ["wilma", "betty", "pebbles"]
        male = ['fred', 'barney', 'mr slate', 'dino']
        image_id = self.image_ids[idx]
        globalIDs = [image_id] + self.followings[image_id]
        ref_flags = []
        captions = []
        images = []
        for idx in range(len(globalIDs)):
            ref_flag = False
            globalID = globalIDs[idx]
            item = self.annotations[globalID]
            caption = item['caption']
            if idx == 0:
                imidiate_char = self.characters[globalID]
            else:
                if sorted(self.characters[globalID]) == sorted(imidiate_char) or self.characters[
                    globalID] in imidiate_char:
                    if len(imidiate_char) > 1:
                        replace_char = "They"
                        if len(imidiate_char) == 2:
                            char_name = imidiate_char[0].capitalize() + " and " + imidiate_char[1].capitalize()
                            if not char_name in caption:
                                char_name = imidiate_char[1].capitalize() + " and " + imidiate_char[0].capitalize()
                        elif len(imidiate_char) == 3:
                            char_name = imidiate_char[0].capitalize() + ", " + imidiate_char[1].capitalize() + " and " + \
                                        imidiate_char[2].capitalize()
                    elif imidiate_char in female or self.characters[globalID] in female:
                        replace_char = "She"
                        char_name = imidiate_char[0].capitalize()
                    else:
                        replace_char = "He"
                        char_name = imidiate_char[0].capitalize()
                    caption = caption.replace(char_name, replace_char)
                    ref_flag = True
                else:
                    imidiate_char = self.characters[globalID]
            ref_flags.append(ref_flag)
            captions.append(caption)
            video_array = np.load(os.path.join(self.root, f'video_frames_sampled_4x/{globalID}.npy'))
            n_frames = video_array.shape[0]
            random_range = random.randrange(n_frames)
            image_array = video_array[random_range]
            img = Image.fromarray(image_array)
            images.append(utils.get_pixel_values_for_model(self.feature_extractor, img))

        select_idx = random.sample(range(1, len(captions)), 1)[0]

        image_id = globalIDs[select_idx]
        caption = ''
        for i in range(select_idx):
            if self.args.interleave:
                caption += ' Image: <img><ImageHere></img> ' + 'Caption: ' + captions[i]
            else:
                caption += ' Caption: ' + captions[i]
        caption += ' Caption: ' + captions[select_idx] + ' Image: '

        for i in range(self.num_tokens):
            caption += f'[IMG{i}]'

        clip_emb = self.clip_embs[image_id].squeeze()
        images = torch.stack(images, dim=0)

        return image_id, images, caption, clip_emb


def find_start_match(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]


class PororoStoryDataset(Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        self.root = args.dataset_dir

        self.feature_extractor = utils.get_feature_extractor_for_model(
            args.visual_model, image_size=args.image_size, train=False)
        self.image_size = args.image_size

        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.precision = args.precision
        self.retrieval_token_idx = args.retrieval_token_idx
        self.gen_token_idx = args.gen_token_idx
        self.num_tokens = args.num_tokens
        self.num_clip_tokens = args.num_clip_tokens

        with open(os.path.join(self.root, 'split.json'), 'r') as f:
            self.splits = json.load(f)

        image_ids = self.splits[split]

        self.followings = pickle.load(open(os.path.join(self.root, 'following_cache3.pkl'), 'rb'))
        self.labels = pickle.load(open(os.path.join(self.root, 'labels.pkl'), 'rb'))

        self.image_ids = [tid for tid in image_ids if tid in self.followings]
        self.annotations = json.load(open(os.path.join(self.root, 'cleaned_annotations.json'), 'r'))
        self.clip_embs = pickle.load(open(os.path.join(self.root, args.clip_emb_file), 'rb'))

        self.font = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        all_characters = ["Pororo", "Loopy", "Crong", "Eddy", "Poby", "Petty", "Tongtong", "Rody", "Harry", "pororo",
                          "loopy", "crong", "eddy", "poby", "petty", "tongtong", "rody", "harry"]
        female = ["Petty", "Loopy", "petty", "loopy"]
        image_id = self.image_ids[idx]
        globalIDs = [image_id] + self.followings[image_id]
        ref_flags = []
        captions = []
        images = []
        for idx in range(len(globalIDs)):
            ref_flag = False
            globalID = globalIDs[idx]
            item = self.annotations[globalID]
            video_array = np.load(os.path.join(self.root, f'video_frames_sampled_4x/{globalID}.npy'))
            n_frames = video_array.shape[0]
            random_range = random.randrange(n_frames)
            image_array = video_array[random_range]
            img = Image.fromarray(image_array)
            images.append(utils.get_pixel_values_for_model(self.feature_extractor, img))
            cap_idx = int(np.ceil((random_range + 1) / n_frames * len(item['captions'])))
            cap_idx = min(cap_idx, len(item['captions']) - 1)
            caption = item["captions"][cap_idx]
            if idx == 0:
                char_name = [x for x in all_characters if x in caption]
                if len(char_name) > 1:
                    if len(caption[:-1].split('.')) > 1:
                        char_name = [x for x in all_characters if x in caption[:-1].split(".")[-1]]
                        if len(char_name) > 0:
                            imidiate_char = char_name[0]
                        else:
                            imidiate_char = ""
                    else:
                        imidiate_char = char_name[0]
                elif len(char_name) == 1:
                    imidiate_char = char_name[0]
                else:
                    imidiate_char = ""
                pre_caption = caption
            else:
                match_substring = find_start_match(pre_caption, caption)
                match_words = match_substring.split(' ')
                for idx, word in enumerate(match_words):
                    if word.replace(",", "").replace("'", "").strip() not in all_characters and not word.__contains__(
                            ',') and not word.__contains__('and'):
                        break
                match_words = match_words[:idx]
                if len(match_words) > 1:
                    replace_string = ' '.join(match_words)
                    caption = caption.replace(replace_string, 'They', 1)
                char_name = [x for x in all_characters if x in caption]
                if len(char_name) > 1:
                    if len(caption[:-1].split('.')) > 1:
                        char_name = [x for x in all_characters if x in caption[:-1].split(".")[-1]]
                        if len(char_name) > 0:
                            char_name = char_name[0]
                        else:
                            char_name = ""
                    else:
                        char_name = char_name[0]
                elif len(char_name) == 1:
                    char_name = char_name[0]
                else:
                    char_name = ""
                if char_name != "" and char_name == imidiate_char:
                    if char_name in female:
                        replace_char = "She"
                        ref_flag = True
                    else:
                        replace_char = "He"
                        ref_flag = True
                    pre_caption = caption
                    caption = caption.replace(char_name, replace_char)
                else:
                    imidiate_char = char_name

            ref_flags.append(ref_flag)
            captions.append(caption)
        select_idx = random.sample(range(1, len(captions)), 1)[0]

        image_id = globalIDs[select_idx]
        caption = ''
        for i in range(select_idx):
            if self.args.interleave:
                caption += ' Image: <img><ImageHere></img> ' + 'Caption: ' + captions[i]
            else:
                caption += ' Caption: ' + captions[i]
        caption += ' Caption: ' + captions[select_idx] + ' Image: '

        for i in range(self.num_tokens):
            caption += f'[IMG{i}]'

        clip_emb = self.clip_embs[image_id].squeeze()
        images = torch.stack(images, dim=0)

        return image_id, images, caption, clip_emb
