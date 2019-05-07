import sys
from mosestokenizer import MosesTokenizer

tokenize = MosesTokenizer('en')

for line in sys.stdin:
    print(" ".join(tokenize(line)))
