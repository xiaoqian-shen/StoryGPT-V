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
import types
import itertools
import os
import pickle
import random

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

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

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
    # check_list=['s_02_e_32_shot_027038_027112','s_01_e_26_shot_009634_009708','s_01_e_27_shot_003202_003276','s_01_e_28_shot_016346_016420','s_03_e_28_shot_020113_020187','s_02_e_31_shot_013970_014044',
    # 's_03_e_28_shot_004400_004474','s_02_e_30_shot_004004_004078','s_01_e_26_shot_032283_032357','s_06_e_24_shot_006609_006683','s_03_e_28_shot_013781_013855','s_02_e_30_shot_019754_019828',
    # 's_04_e_25_shot_023155_023229','s_03_e_28_shot_005907_005981','s_02_e_32_shot_015893_015967','s_01_e_26_shot_012417_012491']
    check_list=['s_03_e_28_shot_005907_005981', 's_01_e_26_shot_032283_032357', 's_01_e_27_shot_003202_003276']
    for image_id in tqdm(check_list):
        following_ids = [image_id]
        following_ids.extend(following_cache[image_id])
        for idx, image_id in enumerate(following_ids):
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
            elif args.ref_image == 'ori':
                char_names = []
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
            else:
                char_names = []
                inserted_tokens = tokens
            prompt=' '.join(inserted_tokens)
            print(args.ref_image, image_id, prompt, flush=True)
            prompt_text_only = prompt.replace(unique_token, "")

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
                
                from fastcomposer.daam import trace, set_seed
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

                        for instance_id in range(args.num_images_per_prompt):
                            images[instance_id].save(
                                os.path.join(
                                    args.output_dir,
                                    f"{image_id}.png",
                                )
                        )

                        if 'attn' in args.output_dir:
                            heat_map = tc.compute_global_heat_map(prompt=prompt_text_only)
                            words = list(set([segment['word'] for segment in segments]))
                            # words = ['They', 'She', 'He']
                            print(prompt_text_only)
                            for word in words:
                                try:
                                    word_heat_maps = heat_map.compute_word_heat_map(word)
                                    for idx, word_heat_map in enumerate(word_heat_maps):
                                        if word == 'wilma' and idx % 2 == 0:
                                            continue
                                        if word == 'wilma':
                                            idx = idx // 2
                                        word_heat_map.plot_overlay(images[0], word_idx=idx, caption=prompt_text_only, out_file=f'{args.output_dir}/{image_id}_{word}_{idx}.png')
                                except:
                                    continue

        # break
if __name__ == "__main__":
    main()
