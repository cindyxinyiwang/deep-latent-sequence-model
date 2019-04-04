

class HParams(object):
  def __init__(self, **args):
    self.pad = "<pad>"
    self.unk = "<unk>"
    self.bos = "<s>"
    # self.eos = "<\s>"
    self.eos = "</s>"
    self.pad_id = 0
    self.unk_id = 1
    self.bos_id = 2
    self.eos_id = 3

    self.batcher = "sent"
    self.batch_size = 32
    self.src_vocab_size = None
    self.trg_vocab_size = None

    self.inf = float("inf")

    for name, value in args.items():
      setattr(self, name, value)
    if hasattr(self, 'train_src_file_list') and type(self.train_src_file_list) == str:
      self.train_src_file_list = self.train_src_file_list.split(',')
    if hasattr(self, 'train_trg_file_list') and type(self.train_trg_file_list) == str:
      self.train_trg_file_list = self.train_trg_file_list.split(',')   
    if hasattr(self, 'src_vocab_list') and type(self.src_vocab_list) == str:
      self.src_vocab_list = self.src_vocab_list.split(',')
    if hasattr(self, 'trg_vocab_list') and type(self.trg_vocab_list) == str:
      self.trg_vocab_list = self.trg_vocab_list.split(',')
    if hasattr(self, 'out_c_list') and type(self.out_c_list) == str:
      self.out_c_list = [int(c) for c in self.out_c_list.split(',')]
    if hasattr(self, 'k_list') and type(self.k_list) == str:
      self.k_list = [int(c) for c in self.k_list.split(',')]
    if hasattr(self, 'pretrained_src_emb_list') and type(self.pretrained_src_emb_list) == str:
      self.pretrained_src_emb_list = self.pretrained_src_emb_list.split(',')
