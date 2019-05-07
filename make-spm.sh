#!/bin/bash

### change the vocab size as you wish 
vocab_size=32000
dataset="form_em"
 
python src/train-spm.py \
  --input=data/$dataset/train.txt \
  --model_prefix=data/$dataset/spm"$vocab_size" \
  --vocab_size="$vocab_size" 

for f in data/$dataset/*.txt; 
do
  python src/run-spm.py \
    --model=data/$dataset/spm"$vocab_size".model \
    < $f \
    > ${f/txt/spm$vocab_size.txt} 
done

# create bpe vocab
python src/get_vocab.py < data/$dataset/train.spm$vocab_size.txt > data/$dataset/text.spm$vocab_size.vocab
