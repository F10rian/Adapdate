from compressai.utils.eval_model.__main__ import load_checkpoint, inference
from tqdm import tqdm
import torch
from adapters.adapter_net import AdapterNet
from adapters.adapter_net2 import AdapterNetAll

def eval_model(arch: str, checkpoint_path, dataset, dataloader):
    #load model
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
    else:
        model = load_checkpoint(arch, True, checkpoint_path)
        model.eval()
    

    bpp_count = 0
    psnr_count = 0

    for i, images in enumerate(tqdm(dataloader)):
        for image in images:
            model.update()
            result = inference(model, image)

            psnr = result.get("psnr-rgb")
            bpp = result.get("bpp")
            bpp_count += bpp
            psnr_count += psnr


    print("Avg bpp: " + str(bpp_count/len(dataset)))
    print("Avg psnr: " + str(psnr_count/len(dataset)))
    return (bpp_count/len(dataset)), (psnr_count/len(dataset))



