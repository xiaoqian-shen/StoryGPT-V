import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
import random
from tqdm import tqdm
import copy
import pickle
from PIL import Image

def prepare_image_token_idx(image_token_mask, max_num_objects):
    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat(
            [
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )

    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    return image_token_idx, image_token_idx_mask

class EvalStoryDataset(object):
    def __init__(
        self,
        tokenizer,
        object_transforms,
        image_token="<|image|>",
        max_num_objects=4,
        device=None,
        root=None,
        ref_image='text',
        story_len=4,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.object_transforms = object_transforms
        self.root=root
        self.tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.max_num_objects = max_num_objects
        self.device = device
        self.ref_image = ref_image

        with open(os.path.join(self.root,'split.json'),'r') as f:
            self.splits = json.load(f)
        
        image_ids = self.splits['test']
        
        if os.path.exists(os.path.join(self.root, 'following_cache' + str(story_len-1) + '.pkl')):
            self.followings = pickle.load(open(os.path.join(self.root, 'following_cache3.pkl'), 'rb'))
        else:
            self.followings = {}
            all_clips = list(self.splits['test'])
            all_clips.sort()
            for idx in range(0, len(all_clips), story_len):
                clip = all_clips[idx]
                season, episode = int(clip.split('_')[1]), int(clip.split('_')[3])
                has_frames = True
                for c in all_clips[idx+1:idx+story_len]:
                    s_c, e_c = int(c.split('_')[1]), int(c.split('_')[3])
                    if s_c != season or e_c != episode:
                        has_frames = False
                        break
                if has_frames:
                    self.followings[clip] = all_clips[idx+1:idx+story_len]
                else:
                    continue
                
        self.labels = pickle.load(open(os.path.join(self.root, 'labels.pkl'),'rb'))

        self.characters = {}
        annotations = json.load(open(os.path.join(self.root, 'flintstones_annotations_v1-0.json'), 'r'))
        for sample in annotations:
            self.characters[sample["globalID"]] = [c["entityLabel"].strip() for c in sample["characters"]]
        
        filtered_followings = {}
        for i, f in self.followings.items():
            if len(f) == story_len - 1:
                filtered_followings[i] = f
            else:
                continue
        self.followings = filtered_followings

        self.image_ids = [tid for tid in image_ids if tid in self.followings]
        self.annotations = json.load(open(os.path.join(self.root, 'cleaned_annotations.json'), 'r'))

    def set_image_ids(self, image_ids=None):
        self.image_ids = image_ids

    def get_data(self, prompt, char_names, image_id):
        return self.prepare_data(prompt, char_names, image_id)

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)

        num_valid_object = 0
        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                num_valid_object += 1
        noun_phrase_end_mask = [False for _ in range(len(input_ids)-num_valid_object)]

        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    def prepare_data(self, caption, prompt_sd, char_names, image_id, ref_flag, captions, image_ids):
        object_pixel_values = []
        for char_name in char_names:
            name = char_name['word']
            reference_image_path = f'data/chars/{name}.jpeg'

            reference_image = self.object_transforms(
                read_image(reference_image_path, mode=ImageReadMode.RGB)
            ).to(self.device)
            object_pixel_values.append(reference_image)
        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            prompt_sd
        )
        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )
        num_objects = image_token_idx_mask.sum().item()

        if len(object_pixel_values) == 0:
            object_pixel_values = torch.zeros((self.max_num_objects,3,224,224)).to(image_token_idx_mask.device)
        else:
            object_pixel_values = torch.stack(
                object_pixel_values
            )  # [max_num_objects, 3, 256, 256]
        
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "num_objects": torch.tensor(num_objects),
            "prompt_text_only": caption,
            "ref_flag": ref_flag,
            "prompt_sd": prompt_sd,
            "image_id": image_id,
            "captions": captions,
            "image_ids": image_ids,
        }
    def prepare_data_batch(self, image_id):
        chars=['fred','wilma','barney','betty','pebbles','mr slate','dino']
        female = ["wilma", "betty", "pebbles"]
        male =['fred','barney','mr slate','dino']

        globalIDs = [image_id] + self.followings[image_id]
        captions = []
        prompts_sd = []
        char_names = []
        ref_flags = []
        return_list = []
        for idx in range(len(globalIDs)):
            ref_flag = False
            globalID = globalIDs[idx]
            item = self.annotations[globalID]
            caption = item['caption']
            if idx == 0:
                imidiate_char = self.characters[globalID]
            else:
                if sorted(self.characters[globalID]) == sorted(imidiate_char) or self.characters[globalID] in imidiate_char:
                    if len(imidiate_char) > 1:
                        replace_char = "They"
                        if len(imidiate_char) == 2:
                            char_name = imidiate_char[0].capitalize()+ " and " + imidiate_char[1].capitalize()
                            if not char_name in caption:
                                char_name = imidiate_char[1].capitalize()+ " and " + imidiate_char[0].capitalize()
                            caption = caption.replace(char_name, replace_char)
                        elif len(imidiate_char) == 3:
                            char_name = imidiate_char[0].capitalize()+ ", "+ imidiate_char[1].capitalize() + " and " + imidiate_char[2].capitalize()
                            caption = caption.replace(char_name, replace_char)
                    elif imidiate_char[0] in female or self.characters[globalID] in female:
                        replace_char = "She"
                        char_name = imidiate_char[0].capitalize()
                        caption = caption.replace(char_name, replace_char)
                    elif imidiate_char[0] in male or self.characters[globalID] in male:
                        replace_char = "He"
                        char_name = imidiate_char[0].capitalize()
                        caption = caption.replace(char_name, replace_char)
                    ref_flag = True
                else:
                    imidiate_char = self.characters[globalID]
                
            captions.append(caption)
            ref_flags.append(ref_flag)

            inserted_tokens = []
            tokens = caption.split(' ')
            char_name = []
            if self.ref_image != 'text':
                for token in tokens:
                    inserted_tokens.append(token)
                    if token.lower() in chars:
                        inserted_tokens.append(self.image_token)
                        char_name.append({'word': token.lower()})
                        ref_flag = False
            prompt=' '.join(inserted_tokens)
            prompts_sd.append(prompt)
            char_names.append(char_name)

            return_list.append(self.prepare_data(captions[idx], prompts_sd[idx], char_names[idx], globalIDs[idx], ref_flag, captions, globalIDs))
        return return_list

def find_start_match(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

class EvalPororoStoryDataset(object):
    def __init__(
        self,
        tokenizer,
        object_transforms,
        image_token="<|image|>",
        max_num_objects=4,
        device=None,
        root=None,
        ref_image='text',
    ) -> None:
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.object_transforms = object_transforms
        self.root=root
        self.tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.max_num_objects = max_num_objects
        self.device = device
        self.ref_image = ref_image

        with open(os.path.join(self.root,'split.json'),'r') as f:
            self.splits = json.load(f)
        
        image_ids = self.splits['test']
        
        self.followings = pickle.load(open(os.path.join(self.root, 'following_cache3.pkl'), 'rb'))
        self.labels = pickle.load(open(os.path.join(self.root, 'labels.pkl'),'rb'))

        self.image_ids = [tid for tid in image_ids if tid in self.followings]
        self.annotations = json.load(open(os.path.join(self.root, 'cleaned_annotations.json'), 'r'))

    def set_image_ids(self, image_ids=None):
        self.image_ids = image_ids

    def get_data(self, prompt, char_names, image_id):
        return self.prepare_data(prompt, char_names, image_id)

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)

        num_valid_object = 0
        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                num_valid_object += 1
        noun_phrase_end_mask = [False for _ in range(len(input_ids)-num_valid_object)]

        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    def prepare_data(self, caption, prompt_sd, char_names, image_id, ref_flag, captions, image_ids):
        object_pixel_values = []
        for char_name in char_names:
            name = char_name['word']
            reference_image_path = f'data/chars/{name}.jpeg'

            reference_image = self.object_transforms(
                read_image(reference_image_path, mode=ImageReadMode.RGB)
            ).to(self.device)
            object_pixel_values.append(reference_image)
        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            prompt_sd
        )
        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )
        num_objects = image_token_idx_mask.sum().item()

        if len(object_pixel_values) == 0:
            object_pixel_values = torch.zeros((self.max_num_objects,3,224,224)).to(image_token_idx_mask.device)
        else:
            object_pixel_values = torch.stack(
                object_pixel_values
            )  # [max_num_objects, 3, 256, 256]
        
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "num_objects": torch.tensor(num_objects),
            "prompt_text_only": caption,
            "ref_flag": ref_flag,
            "prompt_sd": prompt_sd,
            "image_id": image_id,
            "captions": captions,
            "image_ids": image_ids,
        }
    def prepare_data_batch(self, image_id):
        all_characters = ["Pororo", "Loopy", "Crong", "Eddy", "Poby", "Petty", "Tongtong", "Rody", "Harry", "pororo", "loopy", "crong", "eddy", "poby", "petty", "tongtong", "rody", "harry"]
        female = ["Petty", "Loopy", "petty", "loopy"]

        globalIDs = [image_id] + self.followings[image_id]
        captions = []
        prompts_sd = []
        char_names = []
        ref_flags = []
        return_list = []
        for idx in range(len(globalIDs)):
            ref_flag = False
            globalID = globalIDs[idx]
            item = self.annotations[globalID]
            caption = random.sample(item['captions'], 1)[0]
            if idx == 0:
                char_name = [x for x in all_characters if x in caption]
                if len(char_name) > 1:
                    if len(caption[:-1].split('.')) > 1:
                        char_name = [x for x in all_characters if x in caption[:-1].split(".")[-1]]
                        if len(char_name)>0:
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
                for w_idx, word in enumerate(match_words):
                    if word.replace(",", "").replace("'", "").strip() not in all_characters and not word.__contains__(',') and not word.__contains__('and'):
                        break
                match_words = match_words[:w_idx]
                if len(match_words) > 1:
                    replace_string = ' '.join(match_words)
                    caption = caption.replace(replace_string, 'They')
                char_name = [x for x in all_characters if x in caption]
                if len(char_name) > 1:
                    if len(caption[:-1].split('.')) > 1:
                        char_name = [x for x in all_characters if x in caption[:-1].split(".")[-1]]
                        if len(char_name)>0:
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
            captions.append(caption)
            ref_flags.append(ref_flag)
            char_name = []
            prompts_sd.append(random.sample(item['captions'], 1)[0])
            char_names.append(char_name)

            return_list.append(self.prepare_data(captions[idx], prompts_sd[idx], char_names[idx], globalIDs[idx], ref_flag, captions, globalIDs))
            
        return return_list

class EvalDataset(object):
    def __init__(
        self,
        tokenizer,
        object_transforms,
        image_token="<|image|>",
        max_num_objects=4,
        device=None,
        root=None,
        ref_image='text',
    ) -> None:
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.object_transforms = object_transforms
        self.root=root
        self.tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.max_num_objects = max_num_objects
        self.device = device
        self.image_ids = None
        self.ref_image = ref_image

    def set_image_ids(self, image_ids=None):
        self.image_ids = image_ids

    def get_data(self, prompt, char_names, image_id):
        return self.prepare_data(prompt, char_names, image_id)

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)

        num_valid_object = 0
        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                num_valid_object += 1
        noun_phrase_end_mask = [False for _ in range(len(input_ids)-num_valid_object)]

        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    def prepare_data(self, test_caption, char_names, image_id):
        object_pixel_values = []
        if self.ref_image == 'ori':
            reference_image_npy = f'{self.root}/video_frames_sampled_4x/{image_id}.npy'
            video_array=np.load(reference_image_npy)
            image_array=video_array[random.sample(range(len(video_array)),1)[0]]
            for char_name in char_names:
                try:
                    bbox = np.array(char_name["bbox"]) * 4
                    h1, w1, h2, w2 = bbox
                    ref_image_array=torch.tensor(image_array).permute(2,0,1)
                    ref_image_array=ref_image_array[:, w1:w2, h1:h2]
                    reference_image = self.object_transforms(ref_image_array).to(self.device)
                except:
                    name = char_name['word']
                    reference_image_path = f'data/chars/{name}.jpeg'
                    reference_image = self.object_transforms(
                        read_image(reference_image_path, mode=ImageReadMode.RGB)
                    ).to(self.device)
                object_pixel_values.append(reference_image)
        elif self.ref_image == 'same':
            for char_name in char_names:
                name = char_name['word']
                reference_image_path = f'data/chars/{name}.jpeg'

                reference_image = self.object_transforms(
                    read_image(reference_image_path, mode=ImageReadMode.RGB)
                ).to(self.device)
                object_pixel_values.append(reference_image)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            test_caption
        )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()

        if len(object_pixel_values) == 0:
            object_pixel_values = torch.zeros((self.max_num_objects,3,224,224)).to(image_token_idx_mask.device)
        else:
            object_pixel_values = torch.stack(
                object_pixel_values
            )  # [max_num_objects, 3, 256, 256]
        
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "num_objects": torch.tensor(num_objects),
        }

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        max_num_objects=4,
        num_image_tokens=1,
        image_token="<|image|>",
        uncondition_prob=0,
        text_only_prob=0.1,
        split="train",
        min_num_objects=None,
        args=None,
    ):
        self.args = args
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.num_image_tokens = num_image_tokens
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        
        with open(f'{self.root}/following_cache3.pkl','rb') as f:
            self.following_cache = pickle.load(f)
        
        with open(os.path.join(self.root,'split.json'),'r') as f:
            data_split=json.load(f)

        if split == "train":
            self.image_ids = data_split['train']
        elif split == "test":
            self.image_ids = data_split['test']
        else:
            raise ValueError(f"Unknown split {split}")

        self.tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        with open(os.path.join(self.root,'cleaned_annotations.json'),'r') as f:
            self.ann=json.load(f)

        if min_num_objects is not None:
            filtered_image_ids = []

            for image_id in tqdm(self.image_ids):
                info_dict=self.ann[image_id]
                segments = info_dict["segments"]

                if len(segments) >= min_num_objects:
                    filtered_image_ids.append(image_id)
            self.image_ids = filtered_image_ids

    def __len__(self):
        return len(self.image_ids)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, segments, tokens):
        inserted_tokens = copy.deepcopy(tokens)
        for segment in reversed(segments):
            start = segment["start"]
            end = segment["end"]
            inserted_tokens.insert(int(end), self.image_token)
        caption=' '.join(inserted_tokens)
        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in range(len(input_ids)-len(segments))]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length
        
        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def preprocess(self, image, info_dict, segmap, image_id, caption):
        segments = info_dict["segments"]
        try:
            tokens = info_dict['tokens']
        except:
            tokens = caption.split(' ')

        pixel_values, transformed_segmap = self.train_transforms(image, segmap)

        object_pixel_values = []
        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            segments = []
        elif len(segments) <= 1 and prob < self.uncondition_prob + self.text_only_prob:
            segments = []
        if self.text_only_prob == 1:
            segments = []
        if len(segments) > self.max_num_objects:
            segments = random.sample(segments, self.max_num_objects)

        segments = sorted(segments, key=lambda x: x["end"])

        background = self.object_processor.get_background(image)

        for segment in segments:
            id = segment["id"]
            bbox = np.array(segment["bbox"]) * self.args.train_resolution // 128  # [h1, w1, h2, w2]
            object_image = self.object_processor(
                copy.deepcopy(image), background, segmap, id, bbox
            )
            object_pixel_values.append(self.object_transforms(object_image))
            object_segmaps.append(transformed_segmap == id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, segments, tokens
        )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()
        object_pixel_values = object_pixel_values[:num_objects]
        object_segmaps = object_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            object_segmaps
        ).float()  # [max_num_objects, 256, 256]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": torch.tensor(num_objects),
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        video_array = np.load(os.path.join(self.root,f'video_frames_sampled_4x/{image_id}.npy'))
        n_frames=video_array.shape[0]
        random_range = random.randrange(n_frames)
        image_array = video_array[random_range]
        image=torch.from_numpy(image_array).permute(2,0,1)

        info_dict = self.ann[image_id]

        try:
            segmap = torch.from_numpy(np.load(os.path.join(self.root,f'seg_mask_4x_comb/{image_id}-frame{random_range}.npy')))
        except:
            segmap = torch.zeros(image.shape[-2],image.shape[-1])

        if self.device is not None:
            image = image.to(self.device)
            segmap = segmap.to(self.device)
        try:
            caption = info_dict["caption"]
        except:
            idx = int(np.ceil((random_range + 1) / n_frames * len(info_dict["captions"])))
            idx = min(idx, len(info_dict["captions"]) - 1)
            caption = info_dict["captions"][idx]

        return self.preprocess(image, info_dict, segmap, image_id, caption)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])

    image_token_mask = torch.cat([example["image_token_mask"] for example in examples])
    image_token_idx = torch.cat([example["image_token_idx"] for example in examples])
    image_token_idx_mask = torch.cat(
        [example["image_token_idx_mask"] for example in examples]
    )

    object_pixel_values = torch.stack(
        [example["object_pixel_values"] for example in examples]
    )
    object_segmaps = torch.stack([example["object_segmaps"] for example in examples])

    num_objects = torch.stack([example["num_objects"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "image_token_mask": image_token_mask,
        "image_token_idx": image_token_idx,
        "image_token_idx_mask": image_token_idx_mask,
        "object_pixel_values": object_pixel_values,
        "object_segmaps": object_segmaps,
        "num_objects": num_objects,
    }
    


def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )

    return dataloader
