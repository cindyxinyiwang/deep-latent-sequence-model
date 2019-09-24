#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-09-23 Junxian <He>
#
# Distributed under terms of the MIT license.

"""
samples lines from multiple parallel corpus
"""

import argparse
import xlsxwriter
import numpy as np

parser = argparse.ArgumentParser(description="")

parser.add_argument('--files', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--num', type=int, default=200)

args = parser.parse_args()

files = args.files.split(',')

content = [open(file_).readlines() for file_ in files]
length = len(content[0])
order = np.random.permutation(length)

workbook = xlsxwriter.Workbook(args.out)
worksheet = workbook.add_worksheet()

def prepare_header(x):
    return x.split('/')[-1]

def prepare_sample(x):
    return x.strip()

files_head = [prepare_header(x) for x in files]
for k in range(len(files_head)):
    worksheet.write(0, k, files_head[k])

for i in range(args.num):
    id_ = order[i]
    writing = [prepare_sample(x[id_]) for x in content]
    for k in range(len(writing)):
        worksheet.write(i+1, k, writing[k])

workbook.close()
