import nnc
import torch
import os
import argparse
import sys

def compress_decompress(filepath="./", qp=-60):
    path = '/'.join(filepath.split('.')[0].split('/')[:-1])
    if len(filepath.split('.')[0].split('/')) > 1: 
        path += '/'
    model_name = filepath.split('.')[0].split('/')[-1:][0]

    compressed_savefile = path + 'Compressed_' + model_name + str(qp) + '.nnc'
    decompressed_savefile = path + 'Decompressed_' + model_name + str(qp) + '.pth'


    nnc.compress_model(filepath, bitstream_path=compressed_savefile, qp=qp)
    nnc.decompress_model(compressed_savefile, model_path=decompressed_savefile)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for compressing and decompressing a net.")
    parser.add_argument("--model-path", type=str, help="Path of the model to be compressed and decompressed.")
    parser.add_argument("--qp", type=int, default=3, help="Quantisation parameter for the compression.")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    compress_decompress(filepath=args.model_path, qp=args.qp)

if __name__ == "__main__":
    main(sys.argv[1:])