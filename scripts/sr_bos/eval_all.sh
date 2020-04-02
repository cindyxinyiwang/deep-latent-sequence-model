printf "\ntranslate sr -> bos\n"
./scripts/sr_bos/translate_sr_bos.sh $1 data/sr_bos/test_0.spm32000.txt data/sr_bos/test_0.attr transfer_bos.txt

path=$1transfer_bos.txt
./multi-bleu.perl data/sr_bos/test_1.txt < $path


printf "\ntranslate bos -> sr\n"
./scripts/sr_bos/translate_sr_bos.sh $1 data/sr_bos/test_1.spm32000.txt data/sr_bos/test_1.attr transfer_sr.txt

path=$1transfer_sr.txt
./multi-bleu.perl data/sr_bos/test_0.txt < $path
