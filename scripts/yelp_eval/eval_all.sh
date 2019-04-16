export CUDA_VISIBLE_DEVICES=0

# printf "\ntranslate reference test\n"
# ./scripts/yelp_eval/translate_yelp.sh $1 data/yelp/test_li.txt data/yelp/test_li.attr transfer_ref.txt

# printf "\nreference BLEU score\n\n"
# path=$1transfer_ref.txt
# ./multi-bleu.perl data/yelp/test_li_reference.txt < $path

# printf "\ntest reference classification\n\n"
# ./scripts/yelp_eval/yelp_classify_test.sh $path data/yelp/test_li.attr

# printf "\ntranslate test_0\n\n"
# ./scripts/yelp_e3val/translate_yelp.sh $1 data/yelp/test_0.txt data/yelp/test_0.attr transfer_test0.txt
# printf "\ntranslate test_1\n\n"
# ./scripts/yelp_eval/translate_yelp.sh $1 data/yelp/test_1.txt data/yelp/test_1.attr transfer_test1.txt

# path0=$1transfer_test0.txt
# path1=$1transfer_test1.txt
# printf "\nLM test_1\n\n"
# ./scripts/yelp_eval/eval_lm.sh 0 pretrained_lm/yelp_style0/model.pt $path1 data/yelp/test_1.attr
# printf "\nLM test_0\n\n"
# ./scripts/yelp_eval/eval_lm.sh 1 pretrained_lm/yelp_style1/model.pt $path0 data/yelp/test_0.attr

# printf "\ntranslate entire test\n\n"
# ./scripts/yelp_eval/translate_yelp.sh $1 data/yelp/test.txt data/yelp/test.attr transfer_entire_test.txt

path=$1transfer_entire_test.txt
printf "\entire test self BLEU score\n\n"
./multi-bleu.perl data/yelp/test.txt < $path

printf "\nentire test classification\n\n"
./scripts/yelp_eval/yelp_classify_test.sh $path data/yelp/test.attr