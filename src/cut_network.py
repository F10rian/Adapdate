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

from compressai.utils.eval_model.__main__ import inference_entropy_estimation, load_checkpoint, inference

from adapters.adapter_net import AdapterNet
from adapters.adapter_net2 import AdapterNetAll

from adapters.adapter_for_adapter_net import Adapter_for_adapter_net
from adapters.adapter_for_adapter_net2 import Adapter_for_adapter_net2


def cut_network(arch="adapter-net", checkpoint_path="./", v=3, R=2):
    unfreeze_list = []

    small_model = Adapter_for_adapter_net(v=v, R=R)

    if arch == "adapter-net":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        for key in ["network", "state_dict", "model_state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
        model = AdapterNet.from_state_dict(state_dict)
        model.eval()
        small_model = Adapter_for_adapter_net(v=v, R=R)
        unfreeze_list = [
                "g_a.1.conv3", "g_a.1.conv4", "g_a.1.conv5", 
                "g_a.3.conv3", "g_a.3.conv4", "g_a.3.conv5",
                "g_a.5.conv3", "g_a.5.conv4", "g_a.5.conv5",
                "g_s.0.conv3", "g_s.0.conv4", "g_s.0.conv5",
                "g_s.2.conv3", "g_s.2.conv4", "g_s.2.conv5",
                "g_s.4.conv3", "g_s.4.conv4", "g_s.4.conv5",
                "g_s.6.conv3", "g_s.6.conv4", "g_s.6.conv5"
                ]
    elif arch == "adapter-net2":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        for key in ["network", "state_dict", "model_state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
        model = AdapterNetAll.from_state_dict(state_dict)
        model.eval()
        small_model = Adapter_for_adapter_net2(v=v, R=R)
        unfreeze_list = [
                "g_a.0.conv3", "g_a.0.conv4", "g_a.0.conv5", 
                "g_a.1.conv3", "g_a.1.conv4", "g_a.1.conv5", 
                "g_a.2.conv3", "g_a.2.conv4", "g_a.2.conv5", 
                "g_a.3.conv3", "g_a.3.conv4", "g_a.3.conv5",
                "g_a.4.conv3", "g_a.4.conv4", "g_a.4.conv5", 
                "g_a.5.conv3", "g_a.5.conv4", "g_a.5.conv5",
                "g_s.0.conv3", "g_s.0.conv4", "g_s.0.conv5",
                "g_s.1.conv3", "g_s.1.conv4", "g_s.1.conv5",
                "g_s.2.conv3", "g_s.2.conv4", "g_s.2.conv5",
                "g_s.3.conv3", "g_s.3.conv4", "g_s.3.conv5",
                "g_s.4.conv3", "g_s.4.conv4", "g_s.4.conv5",
                "g_s.5.conv3", "g_s.5.conv4", "g_s.5.conv5",
                "g_s.6.conv3", "g_s.6.conv4", "g_s.6.conv5"
                ]
    elif arch == "7":
        model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
        unfreeze_list = [
                "g_s.7", "g_s.6", "g_s.5"
                ]
    elif arch == "ResBlocks":
        model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
        unfreeze_list = [
                "g_a.1", "g_a.3", "g_a.5", "g_s.0", "g_s.2", 
                "g_s.4", "g_s.6"
                ]
    elif arch == "AllBlocks":
        model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
        unfreeze_list = [
                "g_a.0", "g_a.1", "g_a.2", "g_a.3", "g_a.4", 
                "g_a.5", 
                "g_s.0", "g_s.1", "g_s.2", "g_s.3", "g_s.4", 
                "g_s.5", "g_s.6"
                ]
    elif arch == "adapter-net-hyperprior":
        model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
        unfreeze_list = [
                "g_a.1.conv3", "g_a.1.conv4", "g_a.1.conv5", 
                "g_a.3.conv3", "g_a.3.conv4", "g_a.3.conv5",
                "g_a.5.conv3", "g_a.5.conv4", "g_a.5.conv5",
                "g_s.0.conv3", "g_s.0.conv4", "g_s.0.conv5",
                "g_s.2.conv3", "g_s.2.conv4", "g_s.2.conv5",
                "g_s.4.conv3", "g_s.4.conv4", "g_s.4.conv5",
                "g_s.6.conv3", "g_s.6.conv4", "g_s.6.conv5", 
                "h_a", "h_s"
                ]
    else:
        model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
        unfreeze_list = []
        print("No valid arch given!")


    for name1, param1 in model.named_parameters():
        if name1.startswith(tuple(unfreeze_list)):
            name = name1.replace(".", "").replace("weight", ".weight").replace("bias", ".bias")
            for name2, param2 in small_model.named_parameters():
                if name2 == name:
                    param2.data = param1.data

    num_params = 0
    for name, param in small_model.named_parameters():
        num_params+= param.numel()
    print(num_params)

    filepath = checkpoint_path
    path = '/'.join(filepath.split('.')[0].split('/')[:-1])
    path += '/'
    model_name = filepath.split('.')[0].split('/')[-1:][0]


    savefile = path + 'only_adapter_' + model_name + '.pth'

    torch.save(small_model.state_dict(), savefile)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for cutting the adapter weights of a network.")
    parser.add_argument("--model", type=str, default="adapter-net", help="Model architecture (default: %(default)s)")
    parser.add_argument("--model-path", type=str, help="Path of the model to be cut.")
    parser.add_argument("--v", type=int, default=3, help="Set v for the kernelsize v*v in the adapter.")
    parser.add_argument("--R", type=int, default=2, help="Set R for the adapter.")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    cut_network(arch=args.model, checkpoint_path=args.model_path, v=args.v, R=args.R)

if __name__ == "__main__":
    main(sys.argv[1:])