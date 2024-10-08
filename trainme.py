# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from wide_resnet import WideResNet

# import quantization lib
import quant as Qu

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--n_shadows", default=0, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="./exp/cifar10", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--quantize",default=0,type=int)
parser.add_argument("--q_only",default=0,type=int)
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def run():
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    # Dataset
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )
    # datadir = Path().home() / "opt/data/cifar"
    datadir = "./cifar"
    train_ds = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)

    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(train_ds)
    np.random.seed(seed)
    keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
    keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()

    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        wandb.log({"loss": loss_total / len(train_dl)})
        
    savedir = "./target"
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/target_keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/target_model.pt")


    calibrate_set = random_split(train_dl, [0.1,0.9])[0]
    calibrate_dl = DataLoader(calibrate_set, batch_size=32, shuffle=False, num_workers=2, drop_last=True) 
    Qu.runPTQ(m, calibrate_dl, test_dl)
    # Qu.run_int4_quant(m)
    
    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    wandb.log({"acc_test": get_acc(m, test_dl)})

    savedir = "./target"
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/quant_keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/quant_model.pt")

def quantize_only():
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

    datadir = "./cifar"
    test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)

    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    # Model
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.load_state_dict(torch.load("./target/target_model.pt",weights_only=True))
    
    m = m.to(dtype=torch.bfloat16)
    m = m.to(DEVICE)
    Qu.run_int4_quant(m, test_dl)


@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    if args.q_only != 0:
        quantize_only()
    else:
        run()
