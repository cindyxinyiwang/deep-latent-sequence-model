import torch
import torch.nn as nn
import numpy as np


# TODO(junxian): bpe specific noise module to be completed
class NoiseLayer(nn.Module):
    """Add noise to words,
    wrapper class of noise function from FAIR (upon some modification):
    https://github.com/facebookresearch/UnsupervisedMT/blob/master/NMT/src/trainer.py
    """
    def __init__(self, word_blank, word_dropout, word_shuffle,
                 pad_index, blank_index, eos_index, bpe_encode=False):
        """
        Args:
            word_blank (float): blank out probability, 0 to disable
            word_dropout (float): drop out probability, 0 to disable
            word_shuffle (float): should be larger than 1., 0 to disable,
                                  larger value means more shuffling noise
            pad_index (int): the pad index
            blank_index (int): the index used to blank out separate words
        """
        super(NoiseLayer, self).__init__()
        self.blank_prob = word_blank
        self.dropout_prob = word_dropout
        self.shuffle_weight = word_shuffle

        self.pad_index = pad_index
        self.blank_index = blank_index
        self.eos_index = eos_index

    def forward(self, words, lengths):
        """perform shuffle, dropout, and blank operations,
        note that the input is required to have bos_index at the start and
        eos_index at the end
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        if self.blank_prob == 0:
            return x, l
        assert 0 < self.blank_prob < 1

        # define words to blank
        # bos_index = self.bos_index[lang_id]
        # assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.blank_prob
        keep[0] = 1  # do not blank the start sentence symbol

        # # be sure to blank entire words
        # bpe_end = self.bpe_end[lang_id][x]
        # word_idx = bpe_end[::-1].cumsum(0)[::-1]
        # word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(len(l)):
            # assert x[l[i] - 1, i] == eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.blank_index for j, w in enumerate(words)]
            new_s.append(self.eos_index)

            sentences.append(new_s)
        # re-construct input
        x2 = x.new_full((max(l), len(l)), fill_value=self.pad_index)
        for i in range(len(l)):
            x2[:l[i], i].copy_(x.new_tensor(sentences[i]))
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        if self.dropout_prob == 0:
            return x, l
        assert 0 < self.dropout_prob < 1

        # define words to drop
        # bos_index = self.bos_index[lang_id]
        # assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.dropout_prob
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        # bpe_end = self.bpe_end[lang_id][x]
        # word_idx = bpe_end[::-1].cumsum(0)[::-1]
        # word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(len(l)):
            assert x[l[i] - 1, i] == self.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.eos_index)

            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = lengths
        x2 = x.new_full((max(l2), len(l2)), fill_value=self.pad_index)
        for i in range(len(l2)):
            x2[:l2[i], i].copy_(x.new_tensor(sentences[i]))
        return x2, l2

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        Args:
            words (LongTensor): the word ids, (seq_len, batch_size)
            lengths (LongTensor): (batch_size)
        """
        if self.shuffle_weight == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.shuffle_weight, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        # bpe_end = self.bpe_end[lang_id][x]
        # word_idx = bpe_end[::-1].cumsum(0)[::-1]
        # word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.shuffle_weight > 1
        x2 = x.clone()
        for i in range(len(l)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            # scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l
