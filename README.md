<div align="center">

# Experimental comparison of regularization on learning dynamics in deep learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)
[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)

</div>

## Description
This repository contains the code necessary for reproducing the results from my master thesis: "Experimental comparison of regularization on learning dynamics in deep learning".

### Abstract
mixup is a neural network training method that generates new samples by linear interpolation of multiple samples and their labels. In the image domain, the mixup method has a proven record of better generalization ability than the traditional Empirical Risk Minimization method (ERM). At the same time, we lack an intuitive understanding of why mixup is helping. In this work, we attempted to understand better the mixup phenomenon, particularly regarding its impact on the difficulty of decision-making by neural networks. First, we conduct a series of experiments to gather necessary knowledge about the nature of mixup. Next, we make a hypothesis that gives an in-depth understating of why mixup improves generalization.

### Access
To get access to the full PDF version of the thesis, please contact me at: piotr.helm.97@gmail.com

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/piotrhm/master-thesis.git
cd master-thesis

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv
# eg. source activate master-thesis

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
