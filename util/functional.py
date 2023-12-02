import random
import torch
from itertools import pairwise, zip_longest
from collections import Counter
import time
import math
import numpy as np


def grouper(iterable, n, *, incomplete='strict', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')


class TextSampler:
    def __init__(self, corpus, max_len, exponent=0.5):
        self.max_len = max_len
        self.words = [word for line in corpus for word in line.split()]
        unigram_long_text = ''.join(self.words)
        self.unigram_counts = Counter(unigram_long_text)
        self.unigram_counts = {k: len(unigram_long_text) / v ** exponent for k, v in self.unigram_counts.items()}

        bigram_long_text = ' '.join(['', *corpus, ''])
        bigram_long_text = [''.join(pair) for pair in pairwise(bigram_long_text)]
        self.bigram_counts = Counter(bigram_long_text)
        self.bigram_counts = {k: len(bigram_long_text) / v ** exponent for k, v in self.bigram_counts.items()}

        self.words_weights = [self.eval_word(word) for word in self.words]
        self.avg_word_width = sum([len(word) for word in self.words]) / len(self.words)
        self.words_per_line = math.floor(self.max_len / self.avg_word_width)

    def eval_word(self, word):
        bigrams = list(pairwise([' ', *word, ' ']))
        unigram_score = sum([self.unigram_counts[c] for c in word]) / len(word)
        bigram_score = sum([self.bigram_counts[''.join(b)] for b in bigrams]) / len(bigrams)
        return (unigram_score + bigram_score) / 2

    def sample(self, batch_size):
        random_words = random.choices(self.words, weights=self.words_weights, k=self.words_per_line * batch_size)
        bins = [[] for _ in range(batch_size)]
        while random_words:
            word = random_words.pop()
            smallest_bin = np.argmin([sum(len(w) for w in bin) for bin in bins])
            bins[smallest_bin].append(word)
        sampled_lines = [' '.join(line) for line in bins]
        return sampled_lines


class GradSwitch:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model

    @staticmethod
    def _set_grad(model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def __enter__(self):
        self._set_grad(self.model, False)
        self._set_grad(self.target_model, True)
        return self.target_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._set_grad(self.model, True)


class MetricCollector:
    def __init__(self):
        self.data = {}

    def __getitem__(self, item):
        avg = self.data[item]['tot'] / self.data[item]['count']
        return avg

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            assert isinstance(value, (list, tuple))
            assert len(key) == len(value)
            for k, v in zip(key, value):
                self._single_setitem(k, v)
        else:
            self._single_setitem(key, value)

    def _single_setitem(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if key not in self.data:
            self.data[key] = {'tot': 0.0, 'count': 0}
        self.data[key]['tot'] += value
        self.data[key]['count'] += 1

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        assert isinstance(other, MetricCollector)
        new = MetricCollector()
        new.data = self.data | other.data
        return new

    def reset(self):
        self.data = {}

    def print(self, prefix='', suffix=''):
        msg = ' | '.join([f'{k}: {self[k]:.4f}' for k in self.data.keys()])
        print(prefix + msg + suffix)

    def dict(self):
        return {k: self[k] for k in self.data.keys()}

    def pytorch_tensor(self):
        return torch.tensor([[self.data[k]['tot'], self.data[k]['count']] for k in sorted(self.data.keys())])

    def load_pytorch_tensor(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == len(self.data) and tensor.shape[1] == 2
        self.data = {k: {'tot': tot, 'count': count} for k, (tot, count) in zip(sorted(self.data.keys()), tensor.tolist())}
        return self


class Clock:
    def __init__(self, collector=None, key=None, verbose=False):
        self.last = None
        self.collector = collector
        self.key = key
        self.verbose = verbose

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stop()

    def start(self):
        self.last = time.time()

    def stop(self):
        elaps = time.time() - self.last
        if self.collector is not None:
            self.collector[self.key] = elaps
        if self.verbose:
            print(f'Clock: {self.key} {elaps:.4f} sec')


def chunk_list(input_list, chunks):
    chunk_size = len(input_list) // chunks
    assert chunk_size * chunks == len(input_list), f'Cannot split list of size {len(input_list)} into {chunks} chunks'
    chunks = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    return chunks


class TeddyDataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        inputs, module_kwargs = super().scatter(inputs, kwargs, device_ids)
        for idx, el in enumerate(inputs):
            for key, val in el[0].items():
                if isinstance(val, list):
                    el[0][key] = chunk_list(val, len(device_ids))[idx]
        return inputs, module_kwargs


if __name__ == '__main__':
    collector = MetricCollector()

    collector['loss'] = 1
    collector['loss'] = 2
    collector['loss'] = 3

    print(collector['loss'])


class ChunkLoader:
    def __init__(self, loader, chunk_size):
        assert isinstance(loader, torch.utils.data.DataLoader)
        self.loader = loader
        self.loader_iter = iter(loader)
        self.chunk_size = chunk_size

    def __iter__(self):
        for _ in range(self.chunk_size):
            try:
                yield next(self.loader_iter)
            except StopIteration:
                self.loader_iter = iter(self.loader)
                yield next(self.loader_iter)

    def __len__(self):
        return self.chunk_size
