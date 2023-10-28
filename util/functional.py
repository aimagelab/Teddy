import random
import torch
from itertools import pairwise, zip_longest
from collections import Counter
import time


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
    def __init__(self, corpus, words_per_line=32, exponent=0.5):
        self.words_per_line = words_per_line
        self.words = [word for line in corpus for word in line.split()]
        unigram_long_text = ''.join(self.words)
        self.unigram_counts = Counter(unigram_long_text)
        self.unigram_counts = {k: len(unigram_long_text) / v ** exponent for k, v in self.unigram_counts.items()}

        bigram_long_text = ' '.join(['', *corpus, ''])
        bigram_long_text = [''.join(pair) for pair in pairwise(bigram_long_text)]
        self.bigram_counts = Counter(bigram_long_text)
        self.bigram_counts = {k: len(bigram_long_text) / v ** exponent for k, v in self.bigram_counts.items()}

        self.words_weights = [self.eval_word(word) for word in self.words]

    def eval_word(self, word):
        bigrams = list(pairwise([' ', *word, ' ']))
        unigram_score = sum([self.unigram_counts[c] for c in word]) / len(word)
        bigram_score = sum([self.bigram_counts[''.join(b)] for b in bigrams]) / len(bigrams)
        return (unigram_score + bigram_score) / 2

    def sample(self, batch_size):
        random_words = random.choices(self.words, weights=self.words_weights, k=self.words_per_line * batch_size)
        sampled_lines = [' '.join(chunk) for chunk in grouper(random_words, self.words_per_line)]
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
        self.data[key]['tot'] = value
        self.data[key]['count'] = 1

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.data = {}

    def print(self, prefix='', suffix=''):
        msg = ' | '.join([f'{k}: {self[k]:.4f}' for k in self.data.keys()])
        print(prefix + msg + suffix)

    def dict(self):
        return {k: self[k] for k in self.data.keys()}


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


if __name__ == '__main__':
    collector = MetricCollector()

    collector['loss'] = 1
    collector['loss'] = 2
    collector['loss'] = 3

    print(collector['loss'])
