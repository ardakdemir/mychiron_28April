#!/usr/bin/python

import argparse
import sys, os
import logging

from utils import raw
from utils.extract_sig_ref import extract

# generate singals from fast5 files
def fast5toSignal(args):
    FLAGS = args
    FLAGS.input_dir = FLAGS.input
    FLAGS.output_dir = FLAGS.output
    FLAGS.recursive = True

    extract(FLAGS)

# directly training and test for the 2000 dataset


def main():

    parser = argparse.ArgumentParser(prog="UNano", description="Nanopore base-calling by U-net")
    
    parser.add_argument('-i', '--input', required=True, help="fast5 file pathes")
    parser.add_argument('-o', '--output', required=True, help="Output fold")
    parser.add_argument('-m', '--model', required=True, help="model path")
    parser.add_argument('-s', '--start', type=int, default=0, help="Start index of the signal file")
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help="batch size for running")
    parser.add_argument('-l', '--segment_len', type=int, default=300, help="Segment signal length")
    parser.add_argument('-j', '--jump', default=30, help="step size for segment")
    parser.add_argument('-t', '--threads', type=int, default=0, help="Thread number")
    parser.add_argument('-e', '--extension', default='fastq', help="output file type")
    parser.add_argument('-mo', '--mode', default="dna", help="Output mode")
    parser.add_argument('-g', '--gpu', default='-1', help="GPU ID for running")
    parser.add_argument('--test_number', default=None, help="number of test reads")

    args = parser.parse_args(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    fast5toSignal(args)

if __name__ == '__main__':
    print(sys.argv[1:])
    main()
