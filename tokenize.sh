dataset="form_em"

for f in data/$dataset/*.ref*; 
do
  python src/mosestokenize.py \
    < $f \
    > $f.tmp
  mv $f.tmp $f 
done
