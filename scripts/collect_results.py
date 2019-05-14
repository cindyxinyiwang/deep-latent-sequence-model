"""
This script collect all results in a directory

"""

import os
import argparse

parser = argparse.ArgumentParser(description="collect results script")

parser.add_argument('--outdir', type=str, help="an high level experiment dir")

args = parser.parse_args()

best_results = []

with open("{}.results".format(args.outdir.strip("/")), "w") as fout:
    for root, subdirs, files in os.walk(args.outdir):
        for file in files:
            if file == "stdout":
                best_results = []
                print("processing {}".format(os.path.join(root, file)))
                fin = open(os.path.join(root, file))
                line = fin.readline()
                while line:
                    if line.startswith("Eval"):
                        lines_save = []
                        while (not line.startswith("ep=")) and line:
                            lines_save += [line]
                            line = fin.readline()
                        if lines_save[-1].startswith("Saving"):
                            best_results = lines_save
                    # print(line)
                    line = fin.readline()

                fout.write("{}\n".format(os.path.join(root, file)))
                fout.write("".join(best_results))
                fout.write("\n-----------------------------------\n\n")
                fin.close()
