from models.transforms import get_object_transforms
from models.data import EvalDataset
from models.model import StoryModel
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from models.utils import parse_args
from accelerate import Accelerator
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os
from tqdm.auto import tqdm
from models.pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
import types
import itertools
import os
import pickle
import random
from collections import defaultdict

@torch.no_grad()
def clip_emb_img(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    model = StoryModel.from_pretrained(args)

    ckpt_name = "pytorch_model.bin"

    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()

    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    del model

    pipe = pipe.to(accelerator.device)

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    object_transforms = get_object_transforms(args)

    unique_token = "<|image|>"

    import json
    with open(f'{args.dataset_name}/split.json','r') as f:
        split=json.load(f)
    
    data = json.load(open(f'{args.dataset_name}/cleaned_annotations.json','r'))
    
        
    with open(f'{args.dataset_name}/following_cache3.pkl','rb') as f:
        following_cache = pickle.load(f)
    
    demo_dataset = EvalDataset(
            test_reference_folder=args.test_reference_folder,
            tokenizer=tokenizer,
            object_transforms=object_transforms,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            root=args.dataset_name,
            ref_image=args.ref_image
        )

    clip_emb_dict = defaultdict()

    save_name = 'clip_emb_img.pkl'

    if os.path.exists(save_name):
        with open(save_name,'rb') as f:
            clip_emb_dict = pickle.load(f)

    for image_id in tqdm(split['train']+split['val']+split['test']):
        caption=data[image_id]['caption']
        segments=data[image_id]['segments']
        segments = sorted(segments, key=lambda x: x["end"])
        tokens=data[image_id]['tokens']
        chars=['fred','wilma','barney','betty','pebbles','mr slate','dino']
        char_names = []
        if args.ref_image == 'same':
            inserted_tokens = []
            for token in tokens:
                inserted_tokens.append(token)
                if token.lower() in chars:
                    inserted_tokens.append(unique_token)
                    char_names.append({'word': token.lower()})
        elif args.ref_image == 'ori':
            inserted_tokens = tokens.copy()
            for segment in reversed(segments):
                if segment['word'] in chars:
                    char_names.append({'word': segment['word'].lower()})
                else:
                    end = segment['end']
                    inserted_tokens.insert(int(end), unique_token)
                    char_names.append(segment)
            inserted_tokens_new = []
            for idx, token in enumerate(inserted_tokens):
                inserted_tokens_new.append(token)
                if token.lower() in chars:
                    if inserted_tokens[idx+1] != unique_token:
                        inserted_tokens_new.append(unique_token)
                        char_names.append({'word': token.lower()})
            inserted_tokens = inserted_tokens_new
        
        prompt=' '.join(inserted_tokens)
        print(args.ref_image, image_id, prompt, flush=True)
        prompt_text_only = prompt.replace(unique_token, "")

        os.makedirs(args.output_dir, exist_ok=True)

        batch = demo_dataset.get_data(prompt, char_names, image_id)

        input_ids = batch["input_ids"].to(accelerator.device)
        text = tokenizer.batch_decode(input_ids)[0]
        
        # print(input_ids)
        image_token_mask = batch["image_token_mask"].to(accelerator.device)

        # print(image_token_mask)
        all_object_pixel_values = (
            batch["object_pixel_values"].unsqueeze(0).to(accelerator.device)
        )
        num_objects = batch["num_objects"].unsqueeze(0).to(accelerator.device)

        all_object_pixel_values = all_object_pixel_values.to(
            dtype=weight_dtype, device=accelerator.device
        )

        object_pixel_values = all_object_pixel_values  # [:, 0, :, :, :]
        
        if pipe.image_encoder is not None:
            object_embeds = pipe.image_encoder(object_pixel_values)
        else:
            object_embeds = None
            
        encoder_hidden_states = pipe.text_encoder(input_ids)[0]

        encoder_hidden_states = pipe.postfuse_module(
            encoder_hidden_states,
            object_embeds,
            image_token_mask,
            num_objects,
        )

        clip_emb_dict[image_id] = encoder_hidden_states.cpu()
        
    with open(save_name,'wb') as f:
        pickle.dump(clip_emb_dict, f)

@torch.no_grad()
def clip_emb(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    model = StoryModel.from_pretrained(args)

    ckpt_name = "pytorch_model.bin"

    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()

    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    del model

    pipe = pipe.to(accelerator.device)

    import json
    with open(f'{args.dataset_name}/split.json','r') as f:
        split=json.load(f)
    
    data = json.load(open(f'{args.dataset_name}/cleaned_annotations.json','r'))

    clip_emb_dict = defaultdict()

    save_name = 'clip_emb_text.pkl'

    if os.path.exists(save_name):
        with open(save_name,'rb') as f:
            clip_emb_dict = pickle.load(f)
    
    with open(f'{args.dataset_name}/following_cache3.pkl','rb') as f:
        following_cache3 = pickle.load(f)
    
    all_ids = []
    for image_id in following_cache3:
        all_ids.append(image_id)
        all_ids.extend(following_cache3[image_id])
    all_ids.extend(split['train'])
    all_ids.extend(split['val'])
    all_ids.extend(split['test'])
    all_ids = set(all_ids)

    for image_id in tqdm(all_ids):
        try:
            tokens=data[image_id]['tokens']
            caption = ' '.join(tokens)
        except:
            caption = random.sample(data[image_id]['captions'], 1)[0]

        encoder_hidden_states = pipe._encode_prompt(
            caption,
            accelerator.device,
            args.num_images_per_prompt,
            do_classifier_free_guidance=False,
        )

        clip_emb_dict[image_id] = encoder_hidden_states.cpu()
        
    with open(save_name,'wb') as f:
        pickle.dump(clip_emb_dict, f)
    print(len(clip_emb_dict))
    print(len(set(split['train']+split['val']+split['test'])))

if __name__ == "__main__":
    args = parse_args()
    if args.ref_image == 'text':
        clip_emb(args)
    else:
        clip_emb_img(args)
