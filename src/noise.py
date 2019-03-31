import torch
import numpy as np


class SimpleNoise(object):
    """wrapper class of noise function from FAIR (upon some modification):
    https://github.com/facebookresearch/UnsupervisedMT/blob/master/NMT/src/trainer.py
    """
    def __init__(self, word_blank, word_dropout, word_shuffle):
        super(SimpleNoise, self).__init__()
        self.word_blank = word_blank
        self.word_dropout = word_dropout
        self.word_shuffle = word_shuffle
    
    def forward(self, words, lengths, lang_id):
        words, lengths = self.word_shuffle(words, lengths, lang_id)
        words, lengths = self.word_dropout(words, lengths, lang_id)
        words, lengths = self.word_blank(words, lengths, lang_id)
        return words, lengths

    def word_blank(self, x, l, lang_id):
        """
        Randomly blank input words.
        """
        if self.word_blank == 0:
            return x, l
        assert 0 < self.word_blank < 1

        # define words to blank
        bos_index = self.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.blank_index for j, w in enumerate(words)]
            new_s.append(self.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.eos_index
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def word_dropout(self, x, l, lang_id):
        """
        Randomly drop input words.
        """
        if self.word_dropout == 0:
            return x, l
        assert 0 < self.word_dropout < 1

        # define words to drop
        bos_index = self.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.eos_index)
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_shuffle(self, x, l, lang_id):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l
