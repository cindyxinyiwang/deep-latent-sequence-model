

params0={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/form_em/train_0.txt",
  "train_trg_file": "data/form_em/train_0.attr",
  "dev_src_file": "data/form_em/dev_0.txt",
  "dev_trg_file": "data/form_em/dev_0.attr",
  "src_vocab": "data/form_em/text.vocab",
  "trg_vocab": "data/form_em/attr.vocab"
}


params1={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/form_em/train_1.txt",
  "train_trg_file": "data/form_em/train_1.attr",
  "dev_src_file": "data/form_em/dev_1.txt",
  "dev_trg_file": "data/form_em/dev_1.attr",
  "src_vocab": "data/form_em/text.vocab",
  "trg_vocab": "data/form_em/attr.vocab"
}


params_main={
  "lm_style0":"pretrained_lm/form_em_style0/model.pt",
  "lm_style1":"pretrained_lm/form_em_style1/model.pt",
  "eval_cls": True
}
