import sentencepiece as spm
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
#parser.add_argument("--file", type=str)
args = parser.parse_args()

spsrc = spm.SentencePieceProcessor()
spsrc.Load(args.model)
for line in sys.stdin:
  print(" ".join(spsrc.EncodeAsPieces(line.strip())))
