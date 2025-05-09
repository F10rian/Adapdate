import sys
import torch
from compressai.utils.eval_model.__main__ import load_checkpoint

from adapters.adapter_net import AdapterNet
from adapters.adapter_net2 import AdapterNetAll


unfreeze_list = []

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
elif arch == "ResBlocks":
    model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
    unfreeze_list = [
            "g_a.1", "g_a.3", "g_a.5", "g_s.0", "g_s.2", 
            "g_s.4", "g_s.6"
            ]
elif arch == "Frozen7":
    model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
    unfreeze_list = [
            "g_s.7", "g_s.6", "g_s.5"
            ]
elif arch == "AllBlocks":
    model = load_checkpoint("cheng2020-anchor", True, checkpoint_path)
    unfreeze_list = [
            "g_a.0", "g_a.1", "g_a.2", "g_a.3", "g_a.4", 
            "g_a.5", 
            "g_s.0", "g_s.1", "g_s.2", "g_s.3", "g_s.4", 
            "g_s.5", "g_s.6"
            ]
else:
        model = load_checkpoint(arch, True, checkpoint_path)
        unfreeze_list = [
            "g_a.1.conv3", "g_a.1.conv4", "g_a.1.conv5", 
            "g_a.3.conv3", "g_a.3.conv4", "g_a.3.conv5",
            "g_a.5.conv3", "g_a.5.conv4", "g_a.5.conv5",
            "g_s.0.conv3", "g_s.0.conv4", "g_s.0.conv5",
            "g_s.2.conv3", "g_s.2.conv4", "g_s.2.conv5",
            "g_s.4.conv3", "g_s.4.conv4", "g_s.4.conv5",
            "g_s.6.conv3", "g_s.6.conv4", "g_s.6.conv5"
            ]


num_params = 0
for name, param in model.named_parameters():
    if name.startswith(tuple(unfreeze_list)):
        print(name, ": ", param.numel())
        num_params+= param.numel()

print(num_params)