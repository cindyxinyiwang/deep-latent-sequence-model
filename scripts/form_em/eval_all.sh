# export CUDA_VISIBLE_DEVICES=0

printf "\ntranslate reference test\n"
./scripts/translate.sh $1 data/form_em/test.txt data/form_em/test.attr transfer_ref.txt

printf "\nreference BLEU score\n\n"
path=$1transfer_ref.txt
./multi-bleu.perl data/form_em/test.ref < $path

printf "\ntest reference classification\n\n"
./scripts/form_em/form_em_classify_test.sh $path data/form_em/test.attr

printf "\ntranslate test_0\n\n"
./scripts/translate.sh $1 data/form_em/test_0.txt data/form_em/test_0.attr transfer_test0.txt
printf "\ntranslate test_1\n\n"
./scripts/translate.sh $1 data/form_em/test_1.txt data/form_em/test_1.attr transfer_test1.txt

path0=$1transfer_test0.txt
path1=$1transfer_test1.txt
printf "\nLM test_1\n\n"
./scripts/form_em/eval_lm.sh 0 pretrained_lm/form_em_style0/model.pt $path1 data/form_em/test_1.attr
printf "\nLM test_0\n\n"
./scripts/form_em/eval_lm.sh 1 pretrained_lm/form_em_style1/model.pt $path0 data/form_em/test_0.attr

printf "\ntranslate entire test\n\n"
./scripts/translate.sh $1 data/form_em/test.txt data/form_em/test.attr transfer_entire_test.txt

path=$1transfer_entire_test.txt
printf "\entire test self BLEU score\n\n"
./multi-bleu.perl data/form_em/test.txt < $path

printf "\nentire test classification\n\n"
./scripts/form_em/form_em_classify_test.sh $path data/form_em/test.attr
