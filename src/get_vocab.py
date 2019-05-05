import sys

vocab = {}
for line in sys.stdin:
  toks = line.split()
  for t in toks:
    if t not in vocab:
      vocab[t] = 0
    vocab[t] += 1

vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
print("<pad>")
print("<unk>")
print("<s>")
print("</s>")

for w, c in vocab:
  print(w)
