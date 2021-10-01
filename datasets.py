import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import _pickle as cPickle

import os
from typing import List


class SigmorphonData(Dataset):
    def __init__(self, input_fn: str, PAD: str="<P>", START: str="<S>", END: str="<E>", UNK: str = "<UNK>"):
        self.PAD = PAD
        self.START = START
        self.END = END
        self.UNK = UNK
        self.samples = [s for s in self._iter_samples(input_fn)]
        self.char2i, self.i2char = self._make_char_index(self.samples)

    def _iter_samples(self, fn):
        with open(fn, "r") as f:
            for line in f:
                lemma, surface, msd = line.strip().split("\t")
                yield lemma, surface, msd

    def _make_char_index(self, samples: List):
        """Generate Dicts for encoding/decoding chars as unique indices"""
        chars = set([self.PAD, self.START, self.END, self.UNK])
        for lemma, surface, msd in samples:
            [chars.add(c) for c in lemma]
            [chars.add(c) for c in surface]
            [chars.add(t) for t in msd.split(";")]

        char2i = {c: i for i, c in enumerate(sorted(chars))}
        # Set indexing by i for lookup (this could also just be a List..)
        i2char = {i: c for c, i in char2i.items()}

        return char2i, i2char

    def write_index(self, outdir: str, fn: str):
        path = os.path.join(outdir, fn)
        with open(f"{path}_char2i.pkl", "wb") as o:
            cPickle.dump(self.char2i, o)

        with open(f"{path}_i2char.pkl", "wb") as o:
            cPickle.dump(self.i2char, o)


    def load_index(self, outdir: str, fn: str):
        path = os.path.join(outdir, fn)
        with open(f"{path}_char2i.pkl", "rb") as f:
            self.char2i = cPickle.load(f)

        with open(f"{path}_i2char.pkl", "rb") as f:
            self.i2char = cPickle.load(f)


    def encode(self, word: str, msd: str=None, add_start_tag=True):
        """Encode a sequence as a pytorch Tensor of indices"""
        seq = [self.START] if add_start_tag else []
        seq.extend(word)
        if msd is not None:
            seq.extend(msd.split(";"))
        seq.append(self.END)

        seq = [self.char2i.get(c, self.unk_idx) for c in seq]

        return torch.LongTensor(seq)

    def decode(self, words: torch.Tensor, ignore_special=False):
        """Take the tensor of indices, and return a List of chars

        words: B x seq_len
        """
        def ignore(c):
            return c in self.special_idx if ignore_special else False

        decoded = []
        for word in words.numpy():
            decoded.append([self.i2char[c] for c in word if not ignore(c)])

        return decoded

    @property
    def vocab_size(self):
        return len(self.char2i)

    @property
    def pad_idx(self):
        return self.char2i[self.PAD]

    @property
    def start_idx(self):
        return self.char2i[self.START]

    @property
    def end_idx(self):
        return self.char2i[self.END]

    @property
    def unk_idx(self):
        return self.char2i[self.UNK]

    @property
    def special_idx(self):
        return {self.pad_idx, self.start_idx, self.unk_idx, self.end_idx}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        lemma, form, msd = self.samples[index]
        inp = self.encode(word=lemma, msd=msd)
        target = self.encode(word=form, add_start_tag=False)

        return inp, target


class SigmorphonWugData(SigmorphonData):

    def _iter_samples(self, fn):
        with open(fn, "r") as f:
            for line in f:
                # Train data
                try:
                    lemma, surface, msd, orth_lem, orth_surface = line.strip().split("\t")
                    # if "+" in surface:
                    #     continue
                    yield lemma, surface, msd
                # Test judgement data
                except ValueError:
                    try:
                        lemma, surface, msd, rating = line.strip().split("\t")
                        # if "+" in surface:
                        #     continue
                        yield lemma, surface, msd
                    except ValueError:
                        lemma, surface, msd = line.strip().split("\t")
                        # if "+" in surface:
                        #     continue
                        yield lemma, surface, msd

    def _make_char_index(self, samples: List):
        """Generate Dicts for encoding/decoding chars as unique indices"""
        chars = set([self.PAD, self.START, self.END, self.UNK])
        for lemma, surface, msd in samples:
            [chars.add(c) for c in lemma.split()]
            [chars.add(c) for c in surface.split()]
            [chars.add(t) for t in msd.split(";")]

        char2i = {c: i for i, c in enumerate(sorted(chars))}
        # Set indexing by i for lookup (this could also just be a List..)
        i2char = {i: c for c, i in char2i.items()}

        return char2i, i2char

    def encode(self, word: str, msd: str=None, add_start_tag=True):
        """Encode a sequence as a pytorch Tensor of indices

        We assume word is a string of space-delimited IPA unicode symbols"""
        seq = [self.START] if add_start_tag else []
        seq.extend(word.split())
        if msd is not None:
            seq.extend(msd.split(";"))
        seq.append(self.END)

        seq = [self.char2i.get(c, self.unk_idx) for c in seq]

        return torch.LongTensor(seq)


class SigmorphonOrthWugData(SigmorphonData):

    def _iter_samples(self, fn):
        with open(fn, "r") as f:
            for line in f:
                # Train data
                # try:
                #     lemma, surface, msd, orth_lem, orth_surface = line.strip().split("\t")
                #     yield lemma, surface, msd
                # # Test judgement data
                # except ValueError:
                #     lemma, surface, msd, rating = line.strip().split("\t")
                #     yield lemma, surface, msd
                lemma, surface, msd, orth_lem, orth_surface = line.strip().split("\t")
                yield orth_lem, orth_surface, msd


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def pad_collate(self, batch: List):
        """
        batch: List of batch_size tuples of input/output pairs of tensors.
        return: a tensor of all words in 'batch' after padding
        """
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # seperate input/target seqs
        batch_in, batch_out = zip(*batch)

        # pad according to max_len
        max_len_in = max([len(t) for t in batch_in])
        padded_in = [self.pad_tensor(t, max_len_in) for t in batch_in]
        batch_in = torch.stack(padded_in)
        batch_in_mask = (batch_in == self.pad_idx)

        max_len_out = max([len(t) for t in batch_out])
        padded_out = [self.pad_tensor(t, max_len_out) for t in batch_out]
        batch_out = torch.stack(padded_out)

        return batch_in, batch_in_mask, batch_out

    def pad_tensor(self, t: torch.Tensor, max: int):
        # Tuple of prepend indices, append_indices
        p = (0, max-len(t))
        # Add p pad_idx to t
        return F.pad(t, p, "constant", self.pad_idx)

    def __call__(self, batch):
        return self.pad_collate(batch)
