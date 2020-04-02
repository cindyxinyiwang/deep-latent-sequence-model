params0={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/decipher/decipher.train.0",
  "train_trg_file": "data/decipher/train_0.attr",
  "dev_src_file": "data/decipher/decipher.dev.0",
  "dev_trg_file": "data/decipher/dev_0.attr",
  "src_vocab": "data/decipher/text.vocab",
  "trg_vocab": "data/decipher/attr.vocab"
}


params1={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/decipher/decipher.train.1",
  "train_trg_file": "data/decipher/train_1.attr",
  "dev_src_file": "data/decipher/decipher.dev.1",
  "dev_trg_file": "data/decipher/dev_1.attr",
  "src_vocab": "data/decipher/text.vocab",
  "trg_vocab": "data/decipher/attr.vocab"
}

params_main={
  "lm_style0":"pretrained_lm/decipher_style0/model.pt",
  "lm_style1":"pretrained_lm/decipher_style1/model.pt",
  "eval_cls": False
}
