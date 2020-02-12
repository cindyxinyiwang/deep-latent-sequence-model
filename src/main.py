import numpy as np
import argparse
import time
import shutil
import gc
import random
import subprocess
import re
import importlib

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataUtil
from hparams import *
from model import *
from utils import *
from cnn_classify import test, CNNClassify, BiLSTMClassify

if __name__ == "__main__":
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--dataset", type=str, 
    help="dataset name, mainly for naming purpose")

parser.add_argument("--always_save", action="store_true", 
    help="always_save")

parser.add_argument("--residue", action="store_true", 
    help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--layer_norm", action="store_true", 
    help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--src_no_char", action="store_true", 
    help="load an existing model")
parser.add_argument("--trg_no_char", action="store_true", 
    help="load an existing model")
parser.add_argument("--char_gate", action="store_true", 
    help="load an existing model")
parser.add_argument("--shuffle_train", action="store_true", 
    help="load an existing model")
parser.add_argument("--ordered_char_dict", action="store_true", 
    help="load an existing model")
parser.add_argument("--out_c_list", type=str, default=None, 
    help="list of output channels for char cnn emb")
parser.add_argument("--k_list", type=str, default=None, 
    help="list of kernel size for char cnn emb")
parser.add_argument("--highway", action="store_true", 
    help="load an existing model")
parser.add_argument("--n", type=int, default=4, 
    help="ngram n")
parser.add_argument("--single_n", action="store_true", 
    help="ngram n")
parser.add_argument("--bpe_ngram", action="store_true", 
    help="bpe ngram")
parser.add_argument("--uni", action="store_true", 
    help="Gu Universal NMT")
parser.add_argument("--pretrained_src_emb_list", type=str, default=None, 
    help="ngram n")
parser.add_argument("--pretrained_trg_emb", type=str, default=None, 
    help="ngram n")

parser.add_argument("--load_model", action="store_true", 
    help="load an existing model")
parser.add_argument("--load_for_test", action="store_true", 
    help="load an existing model and test")
parser.add_argument("--reset_output_dir", action="store_true", 
    help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="", 
    help="path to output directory")
parser.add_argument("--classifier_dir", type=str, default="", 
    help="directory that stores the classifier model")
parser.add_argument("--log_every", type=int, default=50, 
    help="how many steps to write log")
parser.add_argument("--eval_every", type=int, default=500, 
    help="how many steps to compute valid ppl")
parser.add_argument("--clean_mem_every", type=int, default=10, 
    help="how many steps to clean memory")
parser.add_argument("--eval_bleu", action="store_true", 
    help="if calculate BLEU score for dev set")
parser.add_argument("--beam_size", type=int, default=5, 
    help="beam size for dev BLEU")
parser.add_argument("--poly_norm_m", type=float, default=1, 
    help="beam size for dev BLEU")
parser.add_argument("--ppl_thresh", type=float, default=20, 
    help="beam size for dev BLEU")
parser.add_argument("--max_trans_len", type=int, default=300, 
    help="beam size for dev BLEU")
parser.add_argument("--merge_bpe", action="store_true", 
    help="if calculate BLEU score for dev set")
parser.add_argument("--dev_zero", action="store_true", 
    help="if eval at step 0")

parser.add_argument("--cuda", action="store_true", 
    help="GPU or not")
parser.add_argument("--decode", action="store_true", 
    help="whether to decode only")

parser.add_argument("--max_len", type=int, default=10000, 
    help="maximum len considered on the target side")
parser.add_argument("--n_train_sents", type=int, default=None, 
    help="max number of training sentences to load")

parser.add_argument("--d_word_vec", type=int, default=288, 
    help="size of word and positional embeddings")
parser.add_argument("--d_char_vec", type=int, default=None, 
    help="size of word and positional embeddings")
parser.add_argument("--d_model", type=int, default=288, 
    help="size of hidden states")
parser.add_argument("--d_inner", type=int, default=512, 
    help="hidden dim of position-wise ff")
parser.add_argument("--n_layers", type=int, default=1, 
    help="number of lstm layers")
parser.add_argument("--n_heads", type=int, default=3, 
    help="number of attention heads")
parser.add_argument("--d_k", type=int, default=64, 
    help="size of attention head")
parser.add_argument("--d_v", type=int, default=64, 
    help="size of attention head")
parser.add_argument("--pos_emb_size", type=int, default=None, 
    help="size of trainable pos emb")

parser.add_argument("--train_src_file", type=str, default=None, 
    help="source train file")
parser.add_argument("--train_trg_file", type=str, default=None, 
    help="target train file")
parser.add_argument("--dev_src_file", type=str, default=None, 
    help="source valid file")
parser.add_argument("--dev_trg_file", type=str, default=None, 
    help="target valid file")
parser.add_argument("--dev_trg_ref", type=str, default=None, 
    help="target valid file for reference")
parser.add_argument("--src_vocab", type=str, default=None, 
    help="source vocab file")
parser.add_argument("--trg_vocab", type=str, default=None, 
    help="target vocab file")
parser.add_argument("--test_src_file", type=str, default=None, 
    help="source test file")
parser.add_argument("--test_trg_file", type=str, default=None, 
    help="target test file")
parser.add_argument("--src_char_vocab_from", type=str, default=None, 
    help="source char vocab file")
parser.add_argument("--src_char_vocab_size", type=str, default=None, 
    help="source char vocab size")
parser.add_argument("--trg_char_vocab_from", type=str, default=None, 
    help="target char vocab file")
parser.add_argument("--trg_char_vocab_size", type=str, default=None, 
    help="target char vocab file")
parser.add_argument("--src_vocab_size", type=int, default=None, 
    help="src vocab size")
parser.add_argument("--trg_vocab_size", type=int, default=None, 
    help="trg vocab size")

parser.add_argument("--batch_size", type=int, default=32, 
    help="batch_size")
parser.add_argument("--valid_batch_size", type=int, default=20, 
    help="batch_size")
parser.add_argument("--batcher", type=str, default="sent", 
    help="sent|word. Batch either by number of words or number of sentences")
parser.add_argument("--n_train_steps", type=int, default=100000, 
    help="n_train_steps")
parser.add_argument("--n_train_epochs", type=int, default=0, 
    help="n_train_epochs")
parser.add_argument("--dropout", type=float, default=0., 
    help="probability of dropping")
parser.add_argument("--lr", type=float, default=0.001, 
    help="learning rate")
parser.add_argument("--lr_dec", type=float, default=0.5, 
    help="learning rate decay")
parser.add_argument("--lr_min", type=float, default=0.0001, 
    help="min learning rate")
parser.add_argument("--lr_max", type=float, default=0.001, 
    help="max learning rate")
parser.add_argument("--lr_dec_steps", type=int, default=0, 
    help="cosine delay: learning rate decay steps")

parser.add_argument("--n_warm_ups", type=int, default=0, 
    help="lr warm up steps")
parser.add_argument("--lr_schedule", action="store_true", 
    help="whether to use transformer lr schedule")
parser.add_argument("--clip_grad", type=float, default=5., 
    help="gradient clipping")
parser.add_argument("--l2_reg", type=float, default=0., 
    help="L2 regularization")
parser.add_argument("--patience", type=int, default=-1, 
    help="patience")
parser.add_argument("--eval_end_epoch", action="store_true", 
    help="whether to reload the hparams")

parser.add_argument("--seed", type=int, default=19920206, 
    help="random seed")

parser.add_argument("--init_range", type=float, default=0.1, 
    help="L2 init range")
parser.add_argument("--init_type", type=str, default="uniform", 
    help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

parser.add_argument("--share_emb_softmax", action="store_true", 
    help="weight tieing")
parser.add_argument("--label_smoothing", type=float, default=None, 
    help="label smooth")
parser.add_argument("--reset_hparams", action="store_true", 
    help="whether to reload the hparams")

parser.add_argument("--char_ngram_n", type=int, default=0, 
    help="use char_ngram embedding")
parser.add_argument("--max_char_vocab_size", type=int, default=None, 
    help="char vocab size")

parser.add_argument("--char_input", type=str, default=None, 
    help="[sum|cnn]")
parser.add_argument("--char_comb", type=str, default="add", 
    help="[cat|add]")

parser.add_argument("--char_temp", type=float, default=None, 
    help="temperature to combine word and char emb")

parser.add_argument("--pretrained_model", type=str, default=None, 
    help="location of pretrained model")

parser.add_argument("--src_char_only", action="store_true", 
    help="only use char emb on src")
parser.add_argument("--trg_char_only", action="store_true", 
    help="only use char emb on trg")

parser.add_argument("--model_type", type=str, default="seq2seq", 
    help="[seq2seq|transformer]")
parser.add_argument("--share_emb_and_softmax", action="store_true", 
    help="only use char emb on trg")
parser.add_argument("--transformer_wdrop", action="store_true", 
    help="whether to drop out word embedding of transformer")
parser.add_argument("--transformer_relative_pos", action="store_true", 
    help="whether to use relative positional encoding of transformer")
parser.add_argument("--relative_pos_c", action="store_true", 
    help="whether to use relative positional encoding of transformer")
parser.add_argument("--relative_pos_d", action="store_true", 
    help="whether to use relative positional encoding of transformer")
parser.add_argument("--update_batch", type=int, default="1", 
    help="for how many batches to call backward and optimizer update")
parser.add_argument("--layernorm_eps", type=float, default=1e-9, 
    help="layernorm eps")

# noise parameters
parser.add_argument("--word_blank", type=float, default=0.2, 
    help="blank words probability")
parser.add_argument("--word_dropout", type=float, default=0.2, 
    help="drop words probability")
parser.add_argument("--word_shuffle", type=float, default=1.5, 
    help="shuffle sentence strength")

# balance training objective
parser.add_argument("--anneal_epoch", type=int, default=1,
    help="decrease the weight of autoencoding loss from 1.0 to 0.0 in the first anneal_epoch epoch")

# sampling parameters
parser.add_argument("--temperature", type=float, default=1., 
    help="softmax temperature during training, a small value approx greedy decoding")
parser.add_argument("--gumbel_softmax", action="store_true", 
    help="use gumbel softmax in back-translation")
parser.add_argument("--lm", action="store_true", 
    help="whether including the LM loss")
parser.add_argument("--reconstruct", action="store_true", help=
    "whether perform reconstruction or transfer when validating bleu")
parser.add_argument("--decode_on_y", action="store_true", 
    help="whether to use cond on y at every step when decoding")
parser.add_argument("--attempt_before_decay", type=int, default=0, 
    help="times to try before lr decay")


parser.add_argument("--max_pool_k_size", type=int, default=0, 
    help="max pooling kernel size")
parser.add_argument("--gs_soft", action="store_true", 
    help="soft gumbel softmax if true, otherwise straight-through estimator (gumbel-softmax is always used in the KL loss term)")
parser.add_argument("--klw", type=float, default=1., help="KL loss weight")
parser.add_argument("--bt", action="store_true", help="whether use back translation loss")
parser.add_argument("--bt_stop_grad", action="store_true", 
    help="whether stop gradients through back translation, ignored when gumbel_softmax is false")
parser.add_argument("--avg_len", action="store_true",
    help="whether average over sentence length when computing loss")
parser.add_argument("--dual", action="store_true",
    help="replace KL term with LM log likelihood if true (BT+NLL baseline in the paper)")

args = parser.parse_args()

if args.bpe_ngram: args.n = None

if args.output_dir == "":
    dn = "gs{}".format(args.temperature) if args.gumbel_softmax else "t{}".format(args.temperature)
    lm = "_lm" if args.lm else ""
    decode_y = "_seqy" if args.decode_on_y else ""
    if args.gumbel_softmax or args.lm:
      gs_soft = "_soft" if args.gs_soft else "_hard"
    else:
      gs_soft = ""


    gs_str = "_gs" if args.gumbel_softmax else ""

    bt = "_bt" if args.bt else ""
    bt_stop_grad = "_btsg" if args.bt_stop_grad and args.bt and args.gumbel_softmax else ""
    avg = "_avglen" if args.avg_len else ""
    dual_str = "_dual" if args.dual else ""

    args.output_dir = "outputs_{}{}/{}_wd{}_wb{}_ws{}_an{}_pool{}_klw{}_lr{}_{}{}{}{}{}{}{}{}/".format(args.dataset, dual_str, args.dataset,
        args.word_dropout, args.word_blank, args.word_shuffle, args.anneal_epoch,
        args.max_pool_k_size, args.klw, args.lr, dn, lm, bt, decode_y, gs_str, gs_soft,
        bt_stop_grad, avg)

args.device = torch.device("cuda" if args.cuda else "cpu")

config_file = "config.config_{}".format(args.dataset)
params = importlib.import_module(config_file).params_main
args = argparse.Namespace(**params, **vars(args))

def eval(model, classifier, data, crit, step, hparams, eval_bleu=False,
         valid_batch_size=20, tr_logits=None):
  print("Eval at step {0}. valid_batch_size={1}".format(step, valid_batch_size))

  model.eval()
  #data.reset_valid()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  valid_trans_acc = 0
  n_batches = 0

  valid_total = valid_rule_count = valid_word_count = valid_sents = valid_eos_count = 0
  total_bt_loss = total_noise_loss = total_KL_loss = 0.
  total_bt_elbo_loss = total_kl_elbo_loss = 0.
  valid_word_loss, valid_rule_loss, valid_eos_loss = 0, 0, 0
  total_lm_length = total_trans_length = 0
  ELBO_neg = 0
  valid_bleu = None
  if eval_bleu:
    valid_hyp_file = os.path.join(args.output_dir, "dev.trans_{0}".format(step))
    out_file = open(valid_hyp_file, 'w', encoding='utf-8')

  while True:
    # clear GPU memory
    gc.collect()

    # next batch
    x_valid, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y_valid, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_dev(dev_batch_size=hparams.valid_batch_size)
    #print(x_valid)
    #print(x_mask)
    #print(y_valid)
    #print(y_mask)
    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    x_count -= batch_size
    # word count
    valid_words += x_count
    valid_sents += batch_size

    trans_logits, noise_logits, KL_loss, lm_len, trans_len = model.forward(
      x_valid, x_mask, x_len, x_pos_emb_idxs,
      y_valid, y_mask, y_len, y_pos_emb_idxs,
      y_neg,  y_mask, y_len, eval=True)

    trans_elbo_logits = trans_logits
    noise_elbo_logits = noise_logits
    KL_elbo_loss = KL_loss
    lm_elbo_len = lm_len
    trans_elbo_len = trans_len

    # trans_elbo_logits, noise_elbo_logits, KL_elbo_loss, lm_elbo_len, trans_elbo_len = model.forward(
    #   x_valid, x_mask, x_len, x_pos_emb_idxs,
    #   y_valid, y_mask, y_len, y_pos_emb_idxs,
    #   y_neg,  y_mask, y_len, temperature=1.0, eval=True)

    total_lm_length += lm_len
    total_trans_length += trans_len
    labels = x_valid[:,1:].contiguous().view(-1)
    val_loss, trans_loss, noise_loss, val_acc, val_transfer_acc = \
                get_performance(crit, trans_logits, noise_logits, labels, hparams, x_len)

    _, trans_elbo_loss, _, _, _ = \
                get_performance(crit, trans_elbo_logits, noise_elbo_logits, labels, hparams, x_len)

    if hparams.lm:
        val_loss = val_loss + hparams.klw * KL_loss.sum()
        # val_loss = val_loss + KL_loss.sum()
        total_KL_loss += KL_loss.sum().item()
        total_kl_elbo_loss += KL_elbo_loss.sum().item()

    n_batches += 1
    valid_loss += val_loss.item()
    total_bt_loss += trans_loss.item()
    total_bt_elbo_loss += trans_elbo_loss.item()
    total_noise_loss += noise_loss.item()
    valid_acc += val_acc
    valid_trans_acc += val_transfer_acc
    # print("{0:<5d} / {1:<5d}".format(val_acc.data[0], y_count))
    if end_of_epoch:
      break
  # BLEU eval
  if eval_bleu:
    hyps = []
    dev_batch_size = hparams.valid_batch_size if hparams.beam_size == 1 else 1
    while True:
      gc.collect()
      x_valid, x_mask, x_count, x_len, \
      x_pos_emb_idxs, y_valid, y_mask, \
      y_count, y_len, y_pos_emb_idxs, \
      y_neg, batch_size, end_of_epoch, index = data.next_dev(dev_batch_size=dev_batch_size)
      if hparams.reconstruct:
          y_neg = y_valid
      hs = model.translate(
              x_valid, x_mask, x_len, y_neg, y_mask, y_len, beam_size=args.beam_size, max_len=args.max_trans_len, poly_norm_m=args.poly_norm_m)
      hs = reorder(hs, index)
      hyps.extend(hs)
      if end_of_epoch:
        break
    for h in hyps:
      h_best_words = map(lambda wi: data.src_i2w[wi],
                       filter(lambda wi: wi not in [hparams.bos_id, hparams.eos_id], h))
      if hparams.merge_bpe:
        line = ''.join(h_best_words)
        line = line.replace('▁', ' ')
      else:
        line = ' '.join(h_best_words)
      line = line.strip()
      out_file.write(line + '\n')
      out_file.flush()
    out_file.close()

  if hparams.eval_cls:
    # classify accuracy
    # print("Loading model from '{0}'".format(classifier_file_name))
    # classifier = torch.load(classifier_file_name)
    classifier.eval()
    cur_acc, cur_loss = test(classifier, data, hparams, valid_hyp_file, hparams.dev_trg_file, negate=True)
    print("classifier_acc={}".format(cur_acc))

  bt_ppl = np.exp(total_bt_loss / valid_words)
  noise_ppl = np.exp(total_noise_loss / valid_words)

  log_string = "val_step={0:<6d}".format(step)
  log_string += " total={0:<6.2f}".format(valid_loss / valid_sents)
  log_string += " neg ELBO={0:<6.6f}".format((total_kl_elbo_loss + total_bt_elbo_loss) / valid_sents)
  log_string += " KL={0:<6.2f}".format(total_KL_loss / valid_sents)
  log_string += " bt_ppl={0:<6.2f}".format(bt_ppl)
  log_string += " n_ppl={0:<6.2f}".format(noise_ppl)
  log_string += " n_acc={0:<5.4f}".format(valid_acc / valid_words)
  log_string += " bt_acc={0:<5.4f}".format(valid_trans_acc / valid_words)
  log_string += " lm_len={}".format(total_lm_length / valid_sents)
  log_string += " trans_len={}".format(total_trans_length / valid_sents)
  if eval_bleu:
    out_file.close()
    ref_file = args.dev_trg_ref
    bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file))
    log_string += "\n{}".format(bleu_str)
    bleu_str = bleu_str.split('\n')[-1].strip()
    reg = re.compile("BLEU = ([^,]*).*")
    try:
      valid_bleu = float(reg.match(bleu_str).group(1))
    except:
      valid_bleu = 0.
    log_string += " val_bleu={0:<.2f}".format(valid_bleu)
  print(log_string)
  model.train()
  #exit(0)

  return valid_loss / valid_sents, valid_bleu

def train():
  print(args)
  if args.load_model and (not args.reset_hparams):
    print("load hparams..")
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.load_model = args.load_model
    hparams.n_train_steps = args.n_train_steps
  else:
    hparams = HParams(**vars(args))
    hparams.noise_flag = True

  # build or load model
  print("-" * 80)
  print("Creating model")
  if args.load_model:
    data = DataUtil(hparams=hparams)
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
    if not hasattr(model, 'data'):
      model.data = data
    if not hasattr(model.hparams, 'transformer_wdrop'):
      model.hparams.transformer_wdrop = False

    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    #optim = torch.optim.Adam(trainable_params, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=hparams.l2_reg)
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_val_ppl, best_val_bleu, cur_attempt, cur_decay_attempt, lr = torch.load(extra_file_name)
  else:
    if args.pretrained_model:
      model_name = os.path.join(args.pretrained_model, "model.pt")
      print("Loading model from '{0}'".format(model_name))
      model = torch.load(model_name)

      print("load hparams..")
      hparams_file_name = os.path.join(args.pretrained_model, "hparams.pt")
      reload_hparams = torch.load(hparams_file_name)
      reload_hparams.train_src_file_list = hparams.train_src_file_list
      reload_hparams.train_trg_file_list = hparams.train_trg_file_list
      reload_hparams.dropout = hparams.dropout
      reload_hparams.lr_dec = hparams.lr_dec
      hparams = reload_hparams
      data = DataUtil(hparams=hparams)
      model.data = data
    else:
      data = DataUtil(hparams=hparams)
      if args.model_type == 'seq2seq':
        model = Seq2Seq(hparams=hparams, data=data)
      elif args.model_type == 'transformer':
        model = Transformer(hparams=hparams, data=data)
      else:
        print("Model {} not implemented".format(args.model_type))
        exit(0)
      if args.init_type == "uniform" and not hparams.model_type == "transformer":
        print("initialize uniform with range {}".format(args.init_range))
        for p in model.parameters():
          p.data.uniform_(-args.init_range, args.init_range)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
    #optim = torch.optim.Adam(trainable_params)
    step = 0
    #best_val_ppl = None
    best_val_ppl = 1000
    best_val_bleu = None
    cur_attempt = 0
    cur_decay_attempt = 0
    lr = hparams.lr

  model.set_lm()
  model.to(hparams.device)

  if args.eval_cls:
    classifier_file_name = os.path.join(args.classifier_dir, "model.pt")
    print("Loading model from '{0}'".format(classifier_file_name))
    classifier = torch.load(classifier_file_name).to(hparams.device)
  else:
    classifier = None

  if args.reset_hparams:
    lr = args.lr
  crit = get_criterion(hparams)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  if args.load_for_test:
    val_ppl, val_bleu = eval(model, classifier, data, crit, step, hparams, eval_bleu=args.eval_bleu, valid_batch_size=args.valid_batch_size)
    return

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  target_words = total_loss = total_sents = total_noise_corrects = total_transfer_corrects = 0
  total_bt_loss = total_noise_loss = total_KL_loss = 0.
  target_rules, target_total, target_eos = 0, 0, 0
  total_word_loss, total_rule_loss, total_eos_loss = 0, 0, 0
  total_lm_length = total_trans_length = 0
  model.train()
  #i = 0
  dev_zero = args.dev_zero
  tr_loss, update_batch_size = None, 0
  if hparams.anneal_epoch == 0:
    hparams.noise_weight = 0.
    anneal_rate = 0.
  else:
    hparams.noise_weight = 1.
    if hparams.anneal_epoch == -1:
        anneal_rate = 0.
    else:
        anneal_rate = 1.0 / (data.train_size * args.anneal_epoch // hparams.batch_size)

  hparams.gs_temp = 1.
  while True:
    step += 1
    x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, y_sampled, y_sampled_mask, y_sampled_count, y_sampled_len, y_pos_emb_idxs, batch_size,  eop = data.next_train()
    target_words += (x_count - batch_size)
    total_sents += batch_size
    trans_logits, noise_logits, KL_loss, lm_len, trans_len = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train, y_mask, y_len, y_pos_emb_idxs, y_sampled, y_sampled_mask, y_sampled_len)

    total_lm_length += lm_len
    total_trans_length += trans_len

    # not predicting the start symbol
    labels = x_train[:, 1:].contiguous().view(-1)

    cur_tr_loss, trans_loss, noise_loss, cur_tr_acc, cur_tr_transfer_acc = get_performance(crit, trans_logits,
        noise_logits, labels, hparams, x_len)

    assert(cur_tr_loss.item() > 0)

    if hparams.lm:
        cur_tr_loss = cur_tr_loss + hparams.klw * KL_loss.sum()
        total_KL_loss += KL_loss.sum().item()

    hparams.noise_weight = max(0., hparams.noise_weight - anneal_rate)
    if hparams.noise_weight == 0:
        hparams.noise_flag = False

    # if eop:
    #     hparams.gs_temp = max(0.001, hparams.gs_temp * 0.5)

    total_loss += cur_tr_loss.item()
    total_bt_loss += trans_loss.item()
    total_noise_loss += noise_loss.item()

    total_noise_corrects += cur_tr_acc
    total_transfer_corrects += cur_tr_transfer_acc
    if tr_loss is None:
      tr_loss = cur_tr_loss
    else:
      tr_loss = tr_loss + cur_tr_loss
    update_batch_size += batch_size

    if step % args.update_batch == 0:
      # set learning rate
      if args.lr_schedule:
        s = step / args.update_batch + 1
        lr = pow(hparams.d_model, -0.5) * min(
          pow(s, -0.5), s * pow(hparams.n_warm_ups, -1.5))
        set_lr(optim, lr)
      elif step / args.update_batch < hparams.n_warm_ups:
        base_lr = hparams.lr
        base_lr = base_lr * (step / args.update_batch + 1) / hparams.n_warm_ups
        set_lr(optim, base_lr)
        lr = base_lr
      elif args.lr_dec_steps > 0:
        s = (step / args.update_batch) % args.lr_dec_steps
        lr = args.lr_min + 0.5*(args.lr_max-args.lr_min)*(1+np.cos(s*np.pi/args.lr_dec_steps))
        set_lr(optim, lr)
      tr_loss = tr_loss / update_batch_size
      tr_loss.backward()
      #grad_norm = grad_clip(trainable_params, grad_bound=args.clip_grad)
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
      optim.step()
      optim.zero_grad()
      tr_loss = None
      update_batch_size = 0
    # clean up GPU memory
    if step % args.clean_mem_every == 0:
      gc.collect()
    epoch = step // data.n_train_batches
    if (step / args.update_batch) % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format((step / args.update_batch) / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " total={0:<7.2f}".format(total_loss / total_sents)
      log_string += " neg ELBO={0:<7.2f}".format((total_KL_loss + total_bt_loss) / total_sents)
      log_string += " KL={0:<7.2f}".format(total_KL_loss / total_sents)
      log_string += " |g|={0:<5.2f}".format(grad_norm)

      log_string += " bt_ppl={0:<8.2f}".format(np.exp(total_bt_loss / target_words))
      log_string += " n_ppl={0:<8.2f}".format(np.exp(total_noise_loss / target_words))
      log_string += " n_acc={0:<5.4f}".format(total_noise_corrects / target_words)
      log_string += " bt_acc={0:<5.4f}".format(total_transfer_corrects / target_words)

      # noise weight
      log_string += " nw={:.4f}".format(hparams.noise_weight)

      log_string += " lmlen={}".format(total_lm_length // total_sents)
      log_string += " translen={}".format(total_trans_length // total_sents)
      # log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
      log_string += " t={0:<5.2f}".format(since_start)
      print(log_string)
    if args.eval_end_epoch:
      if eop:
        eval_now = True
      else:
        eval_now = False
    elif (step / args.update_batch) % args.eval_every == 0:
      eval_now = True
    else:
      eval_now = False
    if eval_now:
      # based_on_bleu = args.eval_bleu and best_val_ppl is not None and best_val_ppl <= args.ppl_thresh
      based_on_bleu = False
      if args.dev_zero: based_on_bleu = True
      print("target words: {}".format(target_words))
      with torch.no_grad():
        val_ppl, val_bleu = eval(model, classifier, data, crit, step, hparams, eval_bleu=args.eval_bleu, valid_batch_size=args.valid_batch_size)
      if based_on_bleu:
        if best_val_bleu is None or best_val_bleu <= val_bleu:
          save = True
          best_val_bleu = val_bleu
          cur_attempt = 0
          cur_decay_attempt = 0
        else:
          save = False
      else:
        if best_val_ppl is None or best_val_ppl >= val_ppl:
          save = True
          best_val_ppl = val_ppl
          cur_attempt = 0
          cur_decay_attempt = 0
        else:
          save = False
      if save or args.always_save:
        save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, cur_decay_attempt, lr],
                        model, optim, hparams, args.output_dir)
      elif not args.lr_schedule and step >= hparams.n_warm_ups:
        if cur_decay_attempt >= args.attempt_before_decay:
          if val_ppl >= 2 * best_val_ppl:
            print("reload saved best model !!!")
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.dict")))
            hparams = torch.load(os.path.join(args.output_dir, "hparams.pt"))
          lr = lr * args.lr_dec
          set_lr(optim, lr)
          cur_attempt += 1
          cur_decay_attempt = 0
        else:
          cur_decay_attempt += 1
      # reset counter after eval
      log_start_time = time.time()
      target_words = total_sents = total_noise_corrects = total_transfer_corrects = total_loss = 0
      total_bt_loss = total_noise_loss = total_KL_loss = 0.
      target_rules = target_total = target_eos = 0
      total_word_loss = total_rule_loss = total_eos_loss = 0
      total_lm_length = total_trans_length = 0
    if args.patience >= 0:
      if cur_attempt > args.patience: break
    elif args.n_train_epochs > 0:
      if epoch >= args.n_train_epochs: break
    else:
      if step > args.n_train_steps: break

def main():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  if not os.path.isdir(args.output_dir):
    print("-" * 80)
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  elif args.reset_output_dir:
    print("-" * 80)
    print("Path {} exists. Remove and remake.".format(args.output_dir))
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

  print("-" * 80)
  log_file = os.path.join(args.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  train()

if __name__ == "__main__":
  main()
