# StoryVL

Official PyTorch implementation for 

> **Marrying Latent Diffusion and Language Model for Story Visualization**

## Installation

Environment Setup

```
conda env create -f environment.yaml
```

External Package

```
# Use Lavis BLIP2 for Text-Image alignment evaluation
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
cp -r lavis eval/lavis
```

## Data

We provide the segmentation masks obtained from [SAM](https://github.com/facebookresearch/segment-anything) and upscaled frames by [model](nitro/txt2img-f8-large).

[FlintstonesSV](https://arxiv.org/pdf/1804.03608.pdf): [Google Drive]()

[PororoSV](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_StoryGAN_A_Sequential_Conditional_GAN_for_Story_Visualization_CVPR_2019_paper.pdf): [Google Drive]()

## Inference

## Reference

Related repos [BLIP2](https://github.com/salesforce/LAVIS), [FastComposer](https://github.com/mit-han-lab/fastcomposer), [GILL](https://github.com/kohjingyu/gill), [SAM](https://github.com/facebookresearch/segment-anything)

Baseline codes are from [Story-LDM](https://github.com/ubc-vision/Make-A-Story), [StoryDALL-E](https://github.com/adymaharana/storydalle)
