#!/bin/sh

python train.py experiment=F_cifar10C_resnet18_one_cycle_2.yaml logger=wandb;
python train.py experiment=F_cifar10C_resnet18_mixup_one_cycle_2.yaml logger=wandb;
python train.py experiment=F_cifar10C_resnet18_one_cycle_3.yaml logger=wandb;
python train.py experiment=F_cifar10C_resnet18_mixup_one_cycle_3.yaml logger=wandb;

exit
