## :rocket: Get Started

Environment Setup

```
conda env create -f environment.yaml
conda activate story
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

[FlintstonesSV](https://arxiv.org/pdf/1804.03608.pdf): [[Download]](https://storygpt-v.s3.amazonaws.com/data/flintstones.zip)

[PororoSV](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_StoryGAN_A_Sequential_Conditional_GAN_for_Story_Visualization_CVPR_2019_paper.pdf): [[Download]](https://storygpt-v.s3.amazonaws.com/data/pororo.zip)

## :two: Training

First Stage: Char-LDM

```
bash scripts/train_ldm.sh DATASET
```

Prepare CLIP embedding after first stage

```
bash scripts/clip.sh DATASET CKPT_PATH
```

Second Stage: Align LLM with Char-LDM, you can choose [OPT](https://huggingface.co/facebook/opt-6.7b) or [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat)

```
bash scripts/train_llm_v2.sh DATASET LLM_CKPT 1st_CKPT_PATH
```

## :three: Inference

First prepare finetuned weight of BLIP2 on FlintStonesSV and PororoSV. Finetune BLIP2 by yourself or use our provided finetuned checkpoint `captioner.pth` under each dataset folder: [[BLIP2 FlintStonesSV]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/eval/captioner.pth), [[BLIP2 PororoSV]](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/eval/captioner.pth).

We also provide the pretrained character or background classifier for evaluation. [[FlintStonesSV character]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/eval/classifier_char.pt) [[FlintStonesSV background]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/eval/classifier_bg.pt) [[PororoSV character]](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/eval/classifier_char.pt)

Reproduce results using our model checkpoints:

FlintStonesSV: [[First Stage]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/first-stage/pytorch_model.bin) [[Second Stage (OPT)]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/second-stage-opt.zip) [[Second Stage (Llama2)]](https://storygpt-v.s3.amazonaws.com/checkpoints/flintstones/second-stage-llama2.zip)

PororoSV: [[First Stage]](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/first-stage/pytorch_model.bin) [[Second Stage (OPT)]](https://storygpt-v.s3.amazonaws.com/checkpoints/pororo/second-stage.zip)

To use Llama2, please first download the Llama2 checkpoints from [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat). Then, in the 2nd checkpoints folder we provided, update the "llm_model" field in both `args.json` and `model_args.json` to the path of your local Llama2 folder.

```
# First Stage Evaluation
bash scripts/eval.sh DATASET 1st_CKPT_PATH

# Second Stage Evaluation
bash scripts/eval_llm.sh DATASET 1st_CKPT_PATH 2nd_CKPT_PATH
```

## TODO

- [x] Training code
- [x] Evaluation code
- [x] Finetuned BLIP2 checkpoints for Evaluation
- [x] Model checkpoints

## :closed_book: Reference

Related repos [BLIP2](https://github.com/salesforce/LAVIS), [FastComposer](https://github.com/mit-han-lab/fastcomposer), [GILL](https://github.com/kohjingyu/gill), [SAM](https://github.com/facebookresearch/segment-anything), [DAAM](https://github.com/castorini/daam)

Baseline codes are from [LDM](https://github.com/CompVis/latent-diffusion), [Story-LDM](https://github.com/ubc-vision/Make-A-Story), [StoryDALL-E](https://github.com/adymaharana/storydalle)
