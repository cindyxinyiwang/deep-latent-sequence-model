import sys
import argparse
parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--n", type=int, default=4, help="n gram")
parser.add_argument("--ordered", action="store_true", help="store ngrams by order, not by count")
args = parser.parse_args()

n = args.n
vocab = {}
for line in sys.stdin:
  toks = line.split()
  for w in toks:
    for i in range(len(w)):
      for j in range(i+1, min(i+n, len(w))+1):
        char = w[i:j]
        if args.ordered:
          if char not in vocab:
            vocab[char] = len(vocab)
        else:
          if char not in vocab:
            vocab[char] = 0
          vocab[char] += 1
if args.ordered:
  vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=False)
else:
  vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
print("<unk>")

for w, c in vocab:
  print(w)
