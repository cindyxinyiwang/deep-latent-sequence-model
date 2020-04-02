params0={
  "d_word_vec": 256,
  "d_model": 256,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/sr_bos/train_0.spm32000.txt",
  "train_trg_file": "data/sr_bos/train_0.attr",
  "dev_src_file": "data/sr_bos/dev_0.spm32000.txt",
  "dev_trg_file": "data/sr_bos/dev_0.attr",
  "src_vocab": "data/sr_bos/text.spm32000.vocab",
  "trg_vocab": "data/sr_bos/attr.vocab"
}


params1={
  "d_word_vec": 256,
  "d_model": 256,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/sr_bos/train_1.spm32000.txt",
  "train_trg_file": "data/sr_bos/train_1.attr",
  "dev_src_file": "data/sr_bos/dev_1.spm32000.txt",
  "dev_trg_file": "data/sr_bos/dev_1.attr",
  "src_vocab": "data/sr_bos/text.spm32000.vocab",
  "trg_vocab": "data/sr_bos/attr.vocab"
}

params_main={
  "lm_style0":"pretrained_lm/sr_bos_style0/model.pt",
  "lm_style1":"pretrained_lm/sr_bos_style1/model.pt",
  "eval_cls": False
}