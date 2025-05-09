import sys
import torch
from compressai.utils.eval_model.__main__ import load_checkpoint

from adapters.adapter_net import AdapterNet
from adapters.adapter_net2 import AdapterNetAll

from adapters.adapters_of_adapternet_for_transition import Adapter_for_adapter_net

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