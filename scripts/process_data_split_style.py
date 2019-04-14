"""
This script processes data to split different styles
"""

import os
import argparse

parser = argparse.ArgumentParser(description="process data to split styles")

parser.add_argument('--dataset', type=str, help="dataset name")
parser.add_argument('--prefix', type=str)

args = parser.parse_args()

dir_ = os.path.join("data", args.dataset)

vocab_file = os.path.join(dir_, "attr.vocab")

fin_txt = open(os.path.join(dir_, "{}.txt".format(args.prefix)), "r")
fin_attr = open(os.path.join(dir_, "{}.attr".format(args.prefix)), "r")

fout0_txt = open(os.path.join(dir_, "{}_0.txt".format(args.prefix)), "w")
fout0_attr = open(os.path.join(dir_, "{}_0.attr".format(args.prefix)), "w")

fout1_txt = open(os.path.join(dir_, "{}_1.txt".format(args.prefix)), "w")
fout1_attr = open(os.path.join(dir_, "{}_1.attr".format(args.prefix)), "w")

file_dict0 = {"txt": fout0_txt, "attr": fout0_attr}
file_dict1 = {"txt": fout1_txt, "attr": fout1_attr}

file_list = [file_dict0, file_dict1]

word2id = {}
with open(vocab_file) as fvocab:
  for i, word in enumerate(fvocab):
    word2id[word] = i

for sent, attr in zip(fin_txt, fin_attr):
  attrid = word2id[attr]
  file_list[attrid]["txt"].write(sent)
  file_list[attrid]["attr"].write(attr)


