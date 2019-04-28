params0={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/caption/sentiment.train.0",
  "train_trg_file": "data/caption/train_0.attr",
  "dev_src_file": "data/caption/sentiment.dev.0.txt",
  "dev_trg_file": "data/caption/dev_0.attr",
  "src_vocab": "data/caption/text.vocab",
  "trg_vocab": "data/caption/attr.vocab"
}


params1={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/caption/sentiment.train.1",
  "train_trg_file": "data/caption/train_1.attr",
  "dev_src_file": "data/caption/sentiment.dev.1txt",
  "dev_trg_file": "data/caption/dev_1.attr",
  "src_vocab": "data/caption/text.vocab",
  "trg_vocab": "data/caption/attr.vocab"
}

params_main={
  "lm_style0":"pretrained_lm/caption_style0/model.pt",
  "lm_style1":"pretrained_lm/caption_style1/model.pt",
  "eval_cls": False
}