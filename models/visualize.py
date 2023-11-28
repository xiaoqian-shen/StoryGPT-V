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
import os
from tqdm.auto import tqdm
from models.pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
from models.daam import trace, set_seed
import types
import itertools
import os
import pickle
import random
from collections import defaultdict

@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir,exist_ok=True)
    accelerator.wait_for_everyone()

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
    
    data = json.load(open(f'{args.dataset_name}/cleaned_annotations.json','r'))
    
    demo_dataset = EvalDataset(
            tokenizer=tokenizer,
            object_transforms=object_transforms,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            root=args.dataset_name,
            ref_image=args.ref_image
        )
    check_list=['s_04_e_26_shot_008863_008937','s_03_e_28_shot_042480_042554','s_05_e_22_shot_026387_026461','s_02_e_31_shot_044649_044723',
                's_03_e_23_shot_018865_018939','s_02_e_13_shot_040511_040585','s_02_e_16_shot_029104_029178','s_04_e_15_shot_008125_008199',
                's_02_e_30_shot_004004_004078','s_02_e_14_shot_033088_033162','s_05_e_24_shot_003784_003858','s_05_e_25_shot_004385_004459',
                's_01_e_21_shot_031742_031816','s_04_e_12_shot_020570_020644','s_01_e_23_shot_019129_019203']
    for image_id in tqdm(check_list):
        caption=data[image_id]['caption']
        segments=data[image_id]['segments']
        segments = sorted(segments, key=lambda x: x["end"])
        tokens=data[image_id]['tokens']
        chars=['fred','wilma','barney','betty','pebbles','mr slate','dino']
        char_names = []
        inserted_tokens = tokens.copy()
        if args.ref_image == 'same':
                char_names = []
                inserted_tokens = []
                for token in tokens:
                    inserted_tokens.append(token)
                    if token.lower() in chars:
                        inserted_tokens.append(unique_token)
                        char_names.append({'word': token.lower()})
        elif args.ref_image == 'text':
            char_names = []
            inserted_tokens = tokens
        prompt=' '.join(inserted_tokens)
        print(args.ref_image, image_id, prompt, flush=True)
        prompt_text_only = prompt.replace(unique_token, "")

        batch = demo_dataset.get_data(prompt, char_names, image_id)

        input_ids = batch["input_ids"].to(accelerator.device)
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
            
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]

            encoder_hidden_states_text_only = pipe._encode_prompt(
                prompt_text_only,
                accelerator.device,
                args.num_images_per_prompt,
                do_classifier_free_guidance=False,
            )

            encoder_hidden_states = pipe.postfuse_module(
                encoder_hidden_states,
                object_embeds,
                image_token_mask,
                num_objects,
            )

            cross_attention_kwargs = {}
            
            with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
                with trace(pipe) as tc:
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

                    output_dir = os.path.join(args.output_dir, image_id)
                    os.makedirs(output_dir, exist_ok=True)

                    caption = ' '.join(caption.split(' '))
                    heat_map = tc.compute_global_heat_map(prompt=caption)
                    words = list(set([segment['word'] for segment in segments]))
                    words = caption.split(' ')
                    images[0].save(f'{output_dir}/{image_id}.png')
                    word_dict = defaultdict(int)

                    for word_idx, word in enumerate(words):
                        try:
                            word_heat_maps = heat_map.compute_word_heat_map(word)
                            heat_map_index = word_dict[word] + 1 if word.lower() == 'wilma' else word_dict[word]
                            word_heat_map = word_heat_maps[heat_map_index]
                            word_dict[word] += 1
                            word_heat_map.plot_overlay(images[0], caption=caption, out_file=f'{output_dir}/{image_id}_{word_idx:02d}_{word}.png')
                        except:
                            continue
        # break
if __name__ == "__main__":
    main()
