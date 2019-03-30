import sentencepiece as spm
import os
import time
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", type=str)
parser.add_argument("--input", type=str)
parser.add_argument("--vocab_size", type=int)
args = parser.parse_args()

spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false'.format(args.input, args.model_prefix, args.vocab_size))
