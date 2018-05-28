# Self-Attention GAN
Pytorch implementation of Self-Attention Generative Adversarial Networks (SAGAN).
**Zhang, Han, et al. "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018).**

This repository provides a PyTorch implementation of [SAGAN](https://arxiv.org/abs/1805.08318). Both wgan-gp and wgan-hinge loss are ready, but note that wgan-gp is somehow not compatible with the spectral normalization. remove all the spectral normalization at the model for the adoption of wgan-gp.

Self-attentions are applied to later two layers of both discriminator and generator.

Note that due to the pixel-wise self-attention cost a tune of GPU resource. Batch size of 5~6 available for a single Titan X gpu. Reduce the number of self-attention module for less memory consumption. Removing all the self-attention layers still give you good results, which will be shared shortly.

<p align="center"><img width="100%" src="image/main_model.PNG" /></p>

## Current update status

* [ ] Supervised setting
* [ ] generated image results (under training)
* [x] Unsupervised setting (use no label yet) 
* [x] Applied: [Spectral Normalization](https://arxiv.org/abs/1802.05957), code from [here](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
* [x] Implemented: self-attention module, two-timescale update rule (TTUR), wgan-hinge loss, wgan-gp loss

&nbsp;
&nbsp;

## Results
### CelebA dataset (epoch #)
<p align="center"><img width="100%" src="PNG/celeb_result.png" /></p>

### LSUN church-outdoor dataset (epoch #)
<p align="center"><img width="100%" src="PNG/lsun_result.png" /></p>

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.0](http://pytorch.org/)

&nbsp;

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/heykeetae/SAGAN.git
$ cd Self-Attention-GAN
```

#### 2. Install datasets (CelebA or LSUN)
```bash
$ bash download.sh CelebA
or
$ bash download.sh LSUN
```

#### 3. Train 
##### (i) Train
```bash
$ python python main.py --batch_size 6 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_1
```
&nbsp;
