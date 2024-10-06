# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--savedir", default="./target", type=str)
args = parser.parse_args()


@torch.no_grad()
def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    # Dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )
    # datadir = Path().home() / "opt/data/cifar"
    datadir = "./cifar"
    train_ds = CIFAR10(root=datadir, train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    m1 = models.resnet18(weights=None, num_classes=10)
    m1.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m1.maxpool = nn.Identity()

    m1.load_state_dict(torch.load("./target/target_model.pt"))
    m1.to(DEVICE)
    m1.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(train_dl):
            x = x.to(DEVICE)
            outputs = m1(x)
            logits.append(outputs.cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)

    np.save("./target/target_logits.npy", logits_n)
    
    m1.load_state_dict(torch.load("./target/quant_model.pt"))
    m1.to(DEVICE)
    m1.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(train_dl):
            x = x.to(DEVICE)
            outputs = m1(x)
            logits.append(outputs.cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)

    np.save("./target/quant_logits.npy", logits_n)


if __name__ == "__main__":
    run()
