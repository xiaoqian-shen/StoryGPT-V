# StoryVL

Official PyTorch implementation for the paper

> **Marrying Latent Diffusion and Language Model for Story Visualization**

## :rocket: Get Started

Environment Setup

```
conda env create -f environment.yaml
conda activate StoryVL
```

External Package

```
# Use Lavis BLIP2 for Text-Image alignment evaluation
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
cp -r lavis eval/lavis
```

## :one: Data

We provide the segmentation masks obtained from [SAM](https://github.com/facebookresearch/segment-anything) and upscaled frames by [Super-res model](nitro/txt2img-f8-large).

Download dataset in below links and put them under `data/flintstones` and `data/pororo`

[FlintstonesSV](https://arxiv.org/pdf/1804.03608.pdf): [[Google Drive]]()

[PororoSV](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_StoryGAN_A_Sequential_Conditional_GAN_for_Story_Visualization_CVPR_2019_paper.pdf): [[Google Drive]]()

## :two: Training

First Stage: Latend Diffusion Finetune

```
bash scripts/train_ldm.sh DATASET
```

Prepare CLIP embedding after first stage

```
bash scripts/clip.sh DATASET CKPT_PATH
```

Second Stage: Align LLM with Latent Diffusion

```
bash train_gill.sh DATASET
```

## :three: Inference

First prepare finetuned weight of BLIP2 on FlintStonesSV and PororoSV. Finetune BLIP2 by yourself or use our provided finetuned checkpoint `captioner.pth` under each dataset folder.

Reproduce results using our model checkpoints:

FlintStonesSV: [[First Stage]]() [[Second Stage]]()

PororoSV: [[First Stage]]() [[Second Stage]]()

```
# First Stage Evaluation
bash eval.sh DATASET CKPT_PATH

# Second Stage Evaluation
bash eval_llm.sh DATASET 1st_CKPT 2nd_CKPT
```

## TODO

- [x] Training code
- [x] Evaluation code
- [ ] Finetuned BLIP2 checkpoints for Evaluation
- [ ] Datasets
- [ ] Model checkpoints

## :closed_book: Reference

Related repos [BLIP2](https://github.com/salesforce/LAVIS), [FastComposer](https://github.com/mit-han-lab/fastcomposer), [GILL](https://github.com/kohjingyu/gill), [SAM](https://github.com/facebookresearch/segment-anything), [DAAM](https://github.com/castorini/daam)

Baseline codes are from [LDM](https://github.com/CompVis/latent-diffusion), [Story-LDM](https://github.com/ubc-vision/Make-A-Story), [StoryDALL-E](https://github.com/adymaharana/storydalle)
