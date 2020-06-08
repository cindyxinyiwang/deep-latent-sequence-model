# export CUDA_VISIBLE_DEVICES=0

SCRIPT_ROOT=./scripts
SCRIPT=./scripts/shakespeare
printf "\ntranslate shakespeare to mordern\n"
${SCRIPT_ROOT}/translate.sh $1 data/shakespeare/test_0.txt data/shakespeare/test_0.attr transfer_modern.txt

printf "\nshakespeare -> modern bleu\n\n"
path=$1transfer_modern.txt
./multi-bleu.perl data/shakespeare/test_1.txt < $path

printf "\ntranslate modern to shakespeare\n"
${SCRIPT_ROOT}/translate.sh $1 data/shakespeare/test_1.txt data/shakespeare/test_1.attr transfer_shakespeare.txt

printf "\nmodern -> shakespeare bleu\n\n"
path=$1transfer_shakespeare.txt
./multi-bleu.perl data/shakespeare/test_0.txt < $path

path0=$1transfer_modern.txt
path1=$1transfer_shakespeare.txt
printf "\nLM test_1\n\n"
$SCRIPT/eval_lm.sh 0 pretrained_lm/shakespeare_style0/model.pt $path1 data/shakespeare/test_1.attr
printf "\nLM test_0\n\n"
$SCRIPT/eval_lm.sh 1 pretrained_lm/shakespeare_style1/model.pt $path0 data/shakespeare/test_0.attr

printf "\ntranslate entire test\n\n"
${SCRIPT_ROOT}/translate.sh $1 data/shakespeare/test.txt data/shakespeare/test.attr transfer_entire_test.txt

path=$1transfer_entire_test.txt
printf "\entire test self BLEU score\n\n"
./multi-bleu.perl data/shakespeare/test.txt < $path

printf "\nentire test classification\n\n"
$SCRIPT/shakespeare_classify_test.sh $path data/shakespeare/test.attr
