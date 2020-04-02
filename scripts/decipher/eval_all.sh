printf "\ntranslate cipher test\n"
./scripts/translate.sh $1 data/decipher/decipher.test.1 data/decipher/test_1.attr transfer_plain.txt

path=$1transfer_plain.txt
./multi-bleu.perl data/decipher/decipher.test.0 < $path
