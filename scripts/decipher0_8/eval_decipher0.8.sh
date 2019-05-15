export CUDA_VISIBLE_DEVICES=2

printf "\ntranslate cipher test\n"
./scripts/translate.sh $1 data/yelp_decipher/yelp_decipher0.8/decipher.test.1 data/yelp_decipher/yelp_decipher0.8/test_1.attr transfer_plain.txt

path=$1transfer_plain.txt
./multi-bleu.perl data/yelp_decipher/yelp_decipher0.8/decipher.test.0 < $path
