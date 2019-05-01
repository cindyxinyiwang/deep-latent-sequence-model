#!/bin/bash

### change the vocab size as you wish 
vocab_size=32000
 
python src/train-spm.py \
  --input=data/form_em/train.txt \
  --model_prefix=data/form_em/spm"$vocab_size" \
  --vocab_size="$vocab_size" 

for f in data/form_em/*.txt; 
do
  python src/run-spm.py \
    --model=data/form_em/spm"$vocab_size".model \
    < $f \
    > ${f/txt/txt.spm$vocab_size} 
done
