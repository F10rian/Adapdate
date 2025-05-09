import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
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

from dataset import ImageDataset


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

def train_one_epoch_freeze(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        aux_loss = model.aux_loss()
        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg

def save_checkpoint(state, is_best, filename="checkpoint"):
    save = filename + "_last_epoch.pth"
    torch.save(state, save)
    if is_best:
        filename = filename + ".pth"
        shutil.copyfile(save, filename)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="cheng2020-anchor",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("--train-split", type=str, help="Split of the test dataset")
    parser.add_argument("--val-split", type=str, help="Split of the validation dataset")
    parser.add_argument("--savename", type=str, help="name under which models and data should be stored")
    parser.add_argument("--only-adapter", action="store_true", default=False, help="Should every layer except the adapters bee freezed?")
    parser.add_argument("--unfreeze-list", type=int, default=None, help="Name of the list containing layers that should not be freezed")
    parser.add_argument("--data-path", type=str, help="Path to the dataset")
    parser.add_argument("--v", type=int, default=3, help="Set v for the kernelsize v*v in the adapter.")
    parser.add_argument("--R", type=int, default=2, help="Set R for the adapter.")
    parser.add_argument("--quality", type=int, default=0, help="quality")
    parser.add_argument(
        "-e",
        "--epochs",
        default=70,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def configure_optimizers(net, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": learning_rate},
        "aux": {"type": "Adam", "lr": aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def train_model(args, savename="checkpoint", dataset='./', dataset_name=None, train_dataloader=None, val_dataloader=None, batch_size=16, 
seed=None, cuda=True, save=True, patch_size=(256, 256), learning_rate=1e-4, aux_learning_rate=1e-3, lmbda=1e-2, epochs=70, 
num_workers=4, clip_max_norm=1.0, checkpoint=None, train_only_adapter=False):

    train_only_adapter=args.only_adapter
    savename=args.savename 
    save=args.save
    seed=args.seed
    epochs=args.epochs
    dataset_name='Vimeo-90k'
    num_workers=args.num_workers
    checkpoint=args.checkpoint
    patch_size=args.patch_size
    quality=args.quality

    quality_points = [1e-2, 0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
    lmbda = quality_points[quality]

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"

    if args.model == "adapter-net":
        net = AdapterNet(v=args.v , R=args.R)
    elif args.model == "adapter-net2":
        net = AdapterNetAll(v=args.v , R=args.R)
    else:
        net = image_models[args.model](quality=3)
    net = net.to(device)

    if cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, learning_rate, aux_learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda)
    
    last_epoch = 0
    if checkpoint:  # load from previous checkpoint
        print("Loading", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    
    if train_only_adapter:
        # Freeze all layers
        for name, param in net.named_parameters():
            param.requires_grad = False
        #Unfreeze adapter-layers
        for name, param in net.named_parameters():
            if name.startswith(tuple(unfreeze_list[args.unfreeze_list])):
                param.requires_grad = True

    epochs_own = []
    losses_own = []
    best_loss = float("inf")

    if train_only_adapter:
        last_epoch = 0
    
    for epoch in range(last_epoch, epochs):
        if train_only_adapter:
            train_one_epoch_freeze(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                clip_max_norm,
            )   
        else:
            train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                clip_max_norm,
            )
        loss = test_epoch(epoch, val_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename=savename
            )
        epochs_own.append(epoch)
        losses_own.append(loss.item())

    print("epoch, loss")
    for i in range(len(epochs_own)):
        print("(" + str(epochs_own[i]) + "," + str(losses_own[i]) + ")", end="")
    print("")
    print("epochs: " + str(epochs_own))
    print("losses: " + str(losses_own))

    df = pd.DataFrame({'epochs': epochs_own,
                        'losses': losses_own})
    df.to_csv(savename + ".csv", index=False) 


def get_dataloader(path=None, dataset=None, patch_size=(256, 256), transformer=None, batch_size=16):
    
    if dataset is None:
        dataset = ImageFolder(path, transform=transformer)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=(device == "cuda")
    )
    return dataloader

patch_size=(256, 256)
train_transforms = transforms.Compose(
    [transforms.RandomCrop(patch_size), transforms.ToTensor()]
)
test_transforms = transforms.Compose(
    [transforms.CenterCrop(patch_size), transforms.ToTensor()]
)

unfreeze_list = [[
            "g_a.1.conv3", "g_a.1.conv4", "g_a.1.conv5", 
            "g_a.3.conv3", "g_a.3.conv4", "g_a.3.conv5",
            "g_a.5.conv3", "g_a.5.conv4", "g_a.5.conv5",
            "g_s.0.conv3", "g_s.0.conv4", "g_s.0.conv5",
            "g_s.2.conv3", "g_s.2.conv4", "g_s.2.conv5",
            "g_s.4.conv3", "g_s.4.conv4", "g_s.4.conv5",
            "g_s.6.conv3", "g_s.6.conv4", "g_s.6.conv5"
            ], [
            "g_s.7"    
            ], [
            "g_s.7", "g_s.6.conv2."   
            ], [
            "g_s.7", "g_s.6.conv2.", "g_s.6.conv1"  
            ], [
            "g_s.7", "g_s.6.conv2.", "g_s.6.conv1", "g_s.5.upsample"
            ], [
            "g_s.7", "g_s.6.conv2.", "g_s.6.conv1", "g_s.5.upsample", "g_s.5.igdn"
            ], [
            "g_s.7", "g_s.6.conv2.", "g_s.6.conv1", "g_s.5.upsample", "g_s.5.igdn", "g_s.5.conv"
            ], [
            "g_s.7", "g_s.6", "g_s.5"
            ], [
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
            ], [
            "g_a.1", "g_a.3", "g_a.5", "g_s.0", "g_s.2", 
            "g_s.4", "g_s.6"
            ], [
            "g_a.0", "g_a.1", "g_a.2", "g_a.3", "g_a.4", 
            "g_a.5", 
            "g_s.0", "g_s.1", "g_s.2", "g_s.3", "g_s.4", 
            "g_s.5", "g_s.6", 
            ], [
            "g_a.1.conv3", "g_a.1.conv4", "g_a.1.conv5", 
            "g_a.3.conv3", "g_a.3.conv4", "g_a.3.conv5",
            "g_a.5.conv3", "g_a.5.conv4", "g_a.5.conv5",
            "g_s.0.conv3", "g_s.0.conv4", "g_s.0.conv5",
            "g_s.2.conv3", "g_s.2.conv4", "g_s.2.conv5",
            "g_s.4.conv3", "g_s.4.conv4", "g_s.4.conv5",
            "g_s.6.conv3", "g_s.6.conv4", "g_s.6.conv5", 
            "h_a", "h_s"
            ], [
            "h_a"
            ], [
            "h_s"
            ], [
            "entropy_parameters"
            ], [
            "context_prediction"
            ]]
"""
0: Unfreeze all adapter1
1: Unfreeze last 1 layer
2: Unfreeze last 2 layers
3: Unfreeze last 3 layers
4: Unfreeze last 4 layers
5: Unfreeze last 5 layers
6: Unfreeze last 6 layers
7: Unfreeze last 7 layers
8: Unfreeze all adapter2
9: Unfreze all ResBlocks
10: Unfreeze all ResBlocks, ResBlockUps, ResBlockStrides
"""

def main(argv):

    args = parse_args(argv)

    train_list_path = str(args.data_path) + 'sep_trainlist.txt'
    test_list_path = str(args.data_path) + 'sep_testlist.txt'
    sequence_folder = str(args.data_path) + 'sequences'

    train_dataset = ImageDataset(train_list_path, sequence_folder, train_transforms, args.train_split)
    val_dataset = ImageDataset(train_list_path, sequence_folder, test_transforms, args.val_split)

    train_dataloader = get_dataloader(dataset=train_dataset, patch_size=args.patch_size, transformer=None, batch_size=args.batch_size)
    val_dataloader = get_dataloader(dataset=val_dataset, patch_size=args.patch_size, transformer=None, batch_size=args.batch_size)

    train_model(args, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

if __name__ == "__main__":
    main(sys.argv[1:])