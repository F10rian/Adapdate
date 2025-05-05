import argparse
import compressai
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import csv
import random

from compressai.utils.eval_model.__main__ import inference_entropy_estimation
from compressai.utils.eval_model.__main__ import load_checkpoint

from datahandling import *

from eval_model import eval_model

def eval_model_all(dataset_name="Kodak", arch="adapter-net", model_path="./", datasetpath="./"):
    transform_image = transforms.Compose([transforms.RandomCrop(256, 256), transforms.ToTensor()])
    transform_image_2 = transforms.Compose([transforms.ToTensor()])


    match dataset_name:

        case "Vimeo-90k_septuplet":
            test_list_path = datasetpath + 'sep_testlist.txt'
            sequence_folder = datasetpath + 'sequences'

            dataset = Vimeo90KDataset(test_list_path, sequence_folder, transform_image_2, 7, "test")
            dataloader = get_dataloader(dataset=dataset, transformer=None)
        
        case "Vimeo-90k_septuplet_7":
            test_list_path = datasetpath + 'sep_testlist.txt'
            sequence_folder = datasetpath + 'sequences'

            dataset = Vimeo90KDataset(test_list_path, sequence_folder, transform_image_2, 1, "test")
            dataloader = get_dataloader(dataset=dataset, transformer=None)

        case "Kodak":
            dataset = KodakDataset(str(datasetpath), transform_image_2)
            dataloader = get_dataloader(dataset=dataset, transformer=None, batch_size=1, shuffle=False)



    bpp, psnr = eval_model(arch, model_path, dataset, dataloader)


    bpp = [bpp]
    psnr = [psnr]

    path = '/'.join(model_path.split('.')[0].split('/')[:-1])
    if len(model_path.split('.')[0].split('/')) > 1: 
        path += '/'
    model_name = model_path.split('.')[0].split('/')[-1:][0]


    savefile = path + 'Test_' + model_name + '_on_' + dataset_name + '.csv'
    df = pd.DataFrame({'bpp': bpp, 'psnr': psnr})
    df.to_csv(savefile, index=False) 

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for compressing and decompressing a net.")
    parser.add_argument("--test-split", type=str, default="Kodak", help="Test split to use (default: %(default)s)")
    parser.add_argument("--model", type=str, default="adapter-net", help="Model architecture (default: %(default)s)")
    parser.add_argument("--model-path", type=str, help="Path of the model to be compressed and decompressed.")
    parser.add_argument("--data-path", type=str, help="Path to the data to test the model on.")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    eval_model_all(dataset_name=args.test_split, arch=args.model, model_path=args.model_path, datasetpath=args.data_path)

if __name__ == "__main__":
    main(sys.argv[1:])