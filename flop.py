from models.transforms import get_object_transforms
from models.data import EvalStoryDataset, EvalPororoStoryDataset
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
import time
import wandb

from gill import models
from gill import utils

from eval import eval_cls

from torchvision import transforms

@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    wandb.init(project="test", config=vars(args), entity="xiaoqian-shen", name='ours_'+args.dataset+ '_len' +str(args.story_len))

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
    ).to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    model = StoryModel.from_pretrained(args)
    model.eval()

    ckpt_name = "pytorch_model.bin"

    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    model_dir = args.gill_ckpt
    mm_llm = models.load_gill(model_dir, device=accelerator.device)
    mm_llm.eval()
    g_cuda = torch.Generator(device=accelerator.device).manual_seed(1337)

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

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    if args.dataset == 'flintstones':
        eval_char_cls, input_size = eval_cls.initialize_model(model_name='inception', num_classes=7)
        eval_bg_cls, input_size = eval_cls.initialize_model(model_name='inception', num_classes=323)
        eval_char_cls = eval_char_cls.to(accelerator.device)
        eval_bg_cls = eval_bg_cls.to(accelerator.device)
        eval_char_cls.load_state_dict(torch.load(os.path.join(args.dataset_name, 'classifier_char.pt')))
        eval_bg_cls.load_state_dict(torch.load(os.path.join(args.dataset_name, 'classifier_bg.pt')))
        bg_label = pickle.load(open(os.path.join(args.dataset_name, 'labels_bg.pkl'), 'rb'))['label']
    else:
        eval_char_cls, input_size = eval_cls.initialize_model(model_name='inception', num_classes=9)
        eval_char_cls.load_state_dict(torch.load(os.path.join(args.dataset_name, 'classifier_char.pt')))
        eval_char_cls = eval_char_cls.to(accelerator.device)

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    object_transforms = get_object_transforms(args)

    unique_token = "<|image|>"
    print('ref_image: ', args.ref_image)

    import json

    clip_embs = pickle.load(open(os.path.join(args.dataset_name, 'clip_emb_img_test.pkl'), 'rb')) if args.ret else None
    char_label = pickle.load(open(os.path.join(args.dataset_name, 'labels.pkl'), 'rb'))
    if args.dataset == 'flintstones':
        bg_label = pickle.load(open(os.path.join(args.dataset_name, 'labels_bg.pkl'), 'rb'))['label']

    if args.dataset == 'flintstones':
        demo_dataset = EvalStoryDataset(
            test_reference_folder=args.test_reference_folder,
            tokenizer=tokenizer,
            object_transforms=object_transforms,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            root=args.dataset_name,
            ref_image=args.ref_image,
            story_len=args.story_len
        )
    else:
        demo_dataset = EvalPororoStoryDataset(
            test_reference_folder=args.test_reference_folder,
            tokenizer=tokenizer,
            object_transforms=object_transforms,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            root=args.dataset_name,
            ref_image=args.ref_image
        )

    os.makedirs(args.output_dir, exist_ok=True)
    for image_id in tqdm(demo_dataset.image_ids):
        batchs = demo_dataset.prepare_data_batch(image_id)
        gen_images = []
        start_time= time.time()
        for idx, batch in enumerate(batchs):
            # if not batch['ref_flag']:
            #     continue
            prompt_llm = []
            if idx == 0:
                prompt_llm.append('Caption: ' + batch['captions'][idx] + ' Image: ')
            else:
                for i in range(idx + 1):
                    if i == idx:
                        prompt_llm.append(' Caption: ' + batch['captions'][i] + ' Image: ')
                    else:
                        prompt_llm.append(' Image: <img>')
                        prompt_llm.append(gen_images[i])
                        prompt_llm.append('</img> Caption: ' + batch['captions'][i])

            input_ids = batch["input_ids"].to(accelerator.device)
            text = tokenizer.batch_decode(input_ids)[0]
            image_id = batch['image_id']
            image_token_mask = batch["image_token_mask"].to(accelerator.device)
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

            if idx == 0:
                encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                encoder_hidden_states = pipe.postfuse_module(
                    encoder_hidden_states,
                    object_embeds,
                    image_token_mask,
                    num_objects,
                )
            else:
                encoder_hidden_states = mm_llm.generate_for_images_emb(prompt_llm, num_words=2, gen_scale_factor=100.0,
                                                                       generator=g_cuda, emb_matrix=clip_embs).to(
                    object_embeds.dtype)
                args.start_merge_step = 0

            encoder_hidden_states_text_only = pipe._encode_prompt(
                batch['prompt_text_only'],
                accelerator.device,
                args.num_images_per_prompt,
                do_classifier_free_guidance=False,
            )

            cross_attention_kwargs = {}

            images = pipe.inference(
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=args.inference_steps,
                height=args.generate_height,
                width=args.generate_width,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                cross_attention_kwargs=cross_attention_kwargs,
                prompt_embeds_text_only=encoder_hidden_states_text_only,
                start_merge_step=args.start_merge_step,
            ).images

            gen_images.append(images[0])
        
        stop_time=time.time()
        duration =stop_time - start_time
        hours = duration // 3600
        minutes = (duration - (hours * 3600)) // 60
        seconds = duration - ((hours * 3600) + (minutes * 60))
        msg = f'elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg)

if __name__ == "__main__":
    main()