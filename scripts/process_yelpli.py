"""
This script processes the Yelp dataset used in:
https://github.com/lijuncen/Sentiment-and-Style-Transfer

"""

import os
import subprocess
import random
import torch
import numpy as np

if __name__ == '__main__':
    save_dir = "data/yelp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    shen_dir = "data/yelp_li"

    dev0 = "data/yelp_li/sentiment.dev.0"
    dev1 = "data/yelp_li/sentiment.dev.1"
    test0 = "data/yelp_li/reference.0"
    test1 = "data/yelp_li/reference.1"


    ftrain_t = open(os.path.join(save_dir, "dev_li.txt"), "w")
    ftrain_a = open(os.path.join(save_dir, "dev_li.attr"), "w")

    with open(dev0, "r") as fin:
        for line in fin:
            ftrain_t.write(line)
            ftrain_a.write("negative\n")

    with open(dev1, "r") as fin:
        for line in fin:
            ftrain_t.write(line)
            ftrain_a.write("positive\n")

    ftrain_t.close()
    ftrain_a.close()

    ftrain_t = open(os.path.join(save_dir, "test_li.txt"), "w")
    ftrain_a = open(os.path.join(save_dir, "test_li.attr"), "w")

    fref_t = open(os.path.join(save_dir, "test_li_reference.txt"), "w")

    with open(test0, "r") as fin:
        for line in fin:
            orig, ref = line.strip().split("\t")
            ftrain_t.write(orig + "\n")
            ftrain_a.write("negative\n")
            fref_t.write(ref + "\n")

    with open(test1, "r") as fin:
        for line in fin:
            orig, ref = line.strip().split("\t")
            ftrain_t.write(orig + "\n")
            ftrain_a.write("positive\n")
            fref_t.write(ref + "\n")

    ftrain_t.close()
    ftrain_a.close()

    fref_t.close()
