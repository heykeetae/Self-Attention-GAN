# SAGAN
Pytorch implementation of Self-Attention Generative Adversarial Networks (SAGAN)

This repository provides a PyTorch implementation of [SAGAN](https://arxiv.org/abs/1805.08318). Both wgan-gp and wgan-hinge loss are ready, but note that wgan-gp is somehow not compatible with the spectral normalization. remove all the spectral normalization at the model for the adoption of wgan-gp.

<p align="center"><img width="100%" src="image/main_model.PNG" /></p>

## Current update status

* [ ] Supervised setting
* [ ] Image Results 
* [x] Unsupervised setting (use no label yet) 
* [x] Applied: [Spectral Normalization](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
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
$ cd SAGAN
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
