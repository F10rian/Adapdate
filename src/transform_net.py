import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

from torch.utils.data import Dataset
from PIL import Image
import random
import os
import pandas as pd
import numpy as np

from adapters.adapter_net import AdapterNet
from adapters.adapter_net2 import AdapterNetAll


def configure_optimizers(net, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": learning_rate},
        "aux": {"type": "Adam", "lr": aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def transform(to_net="adapter-net", old_net_path="./", v=3, R=2):
    old_net_path = old_net_path
    if to_net == "adapter-net":
        new_net = AdapterNet(N=128, v=v , R=R)
        savepath = old_net_path.split('.')[0] + "_to_adapter_v" + str(v) + "_R" + str(R) + ".pth"
    elif to_net == "adapter-net2":
        new_net = AdapterNetAll(N=128, v=v , R=R)
        savepath = old_net_path.split('.')[0] + "_to_adapter2_v" + str(v) + "_R" + str(R) + ".pth"
    else:
        new_net = image_models["cheng2020-anchor"](quality=3)
        savepath = old_net_path.split('.')[0] + "_ERROR.pth"
    net = image_models["cheng2020-anchor"](quality=3)


    cuda = True
    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, 0, 0)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    last_epoch = 0

    if old_net_path:  # load from previous checkpoint
            print("Loading", old_net_path)
            checkpoint = torch.load(old_net_path, map_location=device)
            last_epoch = checkpoint["epoch"] + 1
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    for name1, param1 in net.named_parameters():
        for name2, param2 in new_net.named_parameters():
            if name1 == name2:
                param2.data = param1.data


    torch.save({
                        "epoch": last_epoch,
                        "state_dict": new_net.state_dict(),
                        "loss": 0,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }, 
                    savepath
                )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for transforming a cheng2020-anchor net into a adapter-net or adapter-net2.")
    parser.add_argument("--model", type=str, default="adapter-net", help="Model architecture (default: %(default)s)")
    parser.add_argument("--model-path", type=str, help="Name under which models and data should be stored")
    parser.add_argument("--v", type=int, default=3, help="Set v for the kernelsize v*v in the adapter.")
    parser.add_argument("--R", type=int, default=2, help="Set R for the adapter.")
    args = parser.parse_args(argv)
    return args

def main(argv):

    args = parse_args(argv)
    transform(to_net=args.model, old_net_path=args.model_path, v=args.v, R=args.R)

if __name__ == "__main__":
    main(sys.argv[1:])