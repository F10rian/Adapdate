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
import wandb
import random
import os
import pandas as pd
import numpy as np

from compressai.utils.eval_model.__main__ import inference_entropy_estimation, load_checkpoint, inference

from adapters.adapter_net import AdapterNet
from adapters.adapter_net2 import AdapterNetAll

from adapters.adapter_for_adapter_net import Adapter_for_adapter_net

arch = str(sys.argv[1])
checkpoint_path = str(sys.argv[2])

if arch == "adapter-net":
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
    model = AdapterNet.from_state_dict(state_dict)
    model.eval()
elif arch == "adapter-net2":
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
    model = AdapterNetAll.from_state_dict(state_dict)
    model.eval()
elif arch == "only_adapter1":
    model = Adapter_for_adapter_net()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
else:
    model = load_checkpoint(arch, True, checkpoint_path)


for name, param in model.named_parameters():
    print(name, ": ", param.numel())

num_params = sum(p.numel() for p in model.parameters())

print(num_params)