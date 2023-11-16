# LLavis

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

Download dataset and put them under `data/flintstones` and `data/pororo`

[FlintstonesSV](https://arxiv.org/pdf/1804.03608.pdf)

[PororoSV](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_StoryGAN_A_Sequential_Conditional_GAN_for_Story_Visualization_CVPR_2019_paper.pdf)

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

First prepare finetuned weight of BLIP2 on FlintStonesSV and PororoSV. Finetune BLIP2 by yourself or use our provided finetuned checkpoint `captioner.pth` under each dataset folder: [BLIP2 FlintStonesSV](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/eval/captioner.pth), [BLIP2 PororoSV](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/eval/captioner.pth).

Reproduce results using our model checkpoints:

FlintStonesSV: [[First Stage]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/first-stage/pytorch_model.bin) [[Second Stage]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/second-stage.zip)

PororoSV: [[First Stage]](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/first-stage/pytorch_model.bin) [[Second Stage]](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/second-stage.zip)

```
# First Stage Evaluation
bash eval.sh DATASET 1st_CKPT_PATH

# Second Stage Evaluation
bash eval_llm.sh DATASET 1st_CKPT_PATH 2nd_CKPT_PATH
```

## TODO

- [x] Training code
- [x] Evaluation code
- [x] Finetuned BLIP2 checkpoints for Evaluation
- [ ] Model checkpoints
- [ ] Datasets

## :closed_book: Reference

Related repos [BLIP2](https://github.com/salesforce/LAVIS), [FastComposer](https://github.com/mit-han-lab/fastcomposer), [GILL](https://github.com/kohjingyu/gill), [SAM](https://github.com/facebookresearch/segment-anything), [DAAM](https://github.com/castorini/daam)

Baseline codes are from [LDM](https://github.com/CompVis/latent-diffusion), [Story-LDM](https://github.com/ubc-vision/Make-A-Story), [StoryDALL-E](https://github.com/adymaharana/storydalle)
