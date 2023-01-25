#!/bin/sh

python train.py experiment=F_cifar100_resnet18_mixup.yaml seed=323232 logger=wandb;
python train.py experiment=F_cifar100_resnet18.yaml seed=323232 logger=wandb;
# python train.py experiment=F_cifar100_resnet18_mixup.yaml logger=wandb;
# python train.py experiment=F_cifar100_resnet18.yaml logger=wandb;
# python train.py experiment=F_cifar100_resnet18_mixup.yaml seed=4545454 logger=wandb;
# python train.py experiment=F_cifar100_resnet18.yaml seed=4545454 logger=wandb;

exit