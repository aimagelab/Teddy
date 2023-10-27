import torch
import random
import math
from itertools import pairwise, zip_longest
from collections import defaultdict, Counter
from einops import repeat
from torch.nn import functional as F


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


class CheckpointScheduler:
    def __init__(self, model, checkpoint_a, checkpoint_b):
        self.model = model
        self.checkpoints_a = checkpoint_a
        self.checkpoints_b = checkpoint_b

        self.model.load_state_dict(self.checkpoints_a)
        self.model.load_state_dict(self.checkpoints_b)
        self.last_alpha = None

    def _step(self, alpha):
        self.last_alpha = alpha
        for param_name, param in self.model.named_parameters():
            param_a = self.checkpoints_a[param_name]
            param_b = self.checkpoints_b[param_name]
            param.data = param_b.data * alpha + param_a.data * (1 - alpha)

    def step(self):
        return self._step(self.alpha)

    @property
    def alpha(self):
        raise NotImplementedError

    def state_dict(self):
        return {
            'checkpoints_a': self.checkpoints_a,
            'checkpoints_b': self.checkpoints_b,
        }

    def load_state_dict(self, state_dict):
        self.checkpoints_a = state_dict['checkpoints_a']
        self.checkpoints_b = state_dict['checkpoints_b']


class RandCheckpointScheduler(CheckpointScheduler):
    def __init__(self, model, checkpoint_a, checkpoint_b):
        super(RandCheckpointScheduler, self).__init__(model, checkpoint_a, checkpoint_b)

    @property
    def alpha(self):
        return random.random()


class SineCheckpointScheduler(CheckpointScheduler):
    def __init__(self, model, checkpoint_a, checkpoint_b, period=100):
        super(SineCheckpointScheduler, self).__init__(model, checkpoint_a, checkpoint_b)
        self.period = period
        self.counter = 0

    def step(self):
        self._step(self.alpha)
        self.counter += 1

    @property
    def alpha(self):
        return (math.sin((self.counter % (self.period * 2)) / self.period * math.pi) + 1) / 2

    def state_dict(self):
        state_dict = super(SineCheckpointScheduler, self).state_dict()
        state_dict['period'] = self.period
        state_dict['counter'] = self.counter
        return state_dict

    def load_state_dict(self, state_dict):
        super(SineCheckpointScheduler, self).load_state_dict(state_dict)
        self.period = state_dict['period']
        self.counter = state_dict['counter']


class LinearScheduler(CheckpointScheduler):
    def __init__(self, model, checkpoint_a, checkpoint_b, period=100):
        super(LinearScheduler, self).__init__(model, checkpoint_a, checkpoint_b)
        self.period = period
        self.counter = 0

    def step(self):
        self._step(self.alpha)
        self.counter += 1

    @property
    def alpha(self):
        return (self.counter % self.period) / self.period

    def state_dict(self):
        state_dict = super(SineCheckpointScheduler, self).state_dict()
        state_dict['period'] = self.period
        state_dict['counter'] = self.counter
        return state_dict

    def load_state_dict(self, state_dict):
        super(SineCheckpointScheduler, self).load_state_dict(state_dict)
        self.period = state_dict['period']
        self.counter = state_dict['counter']


class OneLinearScheduler(CheckpointScheduler):
    def __init__(self, model, checkpoint_a, checkpoint_b, period=100):
        super(OneLinearScheduler, self).__init__(model, checkpoint_a, checkpoint_b)
        self.period = period
        self.counter = 0

    def step(self):
        self._step(self.alpha)
        self.counter += 1

    @property
    def alpha(self):
        return min(self.counter, self.period) / self.period


class RandomLinearScheduler(OneLinearScheduler):
    def __init__(self, model, checkpoint_b, period=100):
        checkpoint_a = {k: torch.rand_like(v) for k, v in checkpoint_b.items()}
        super(RandomLinearScheduler, self).__init__(model, checkpoint_a, checkpoint_b, period)


class AlternatingScheduler:
    def __init__(self, model, *checkpoints):
        self.model = model
        self.checkpoints = checkpoints
        self.counter = 0
        self.last_alpha = None

    def _step(self, idx):
        self.last_alpha = idx
        checkpoint_to_load = self.checkpoints[idx]
        self.model.load_state_dict(checkpoint_to_load)

    def step(self):
        self._step(self.alpha)
        self.counter += 1

    @property
    def alpha(self):
        return self.counter % len(self.checkpoints)


class RandReducingScheduler(CheckpointScheduler):
    def __init__(self, model, checkpoint_a, checkpoint_b, max_certanty=0.9, decay_step=1e-5):
        super(RandReducingScheduler, self).__init__(model, checkpoint_a, checkpoint_b)
        self.max_certanty = max_certanty
        self.decay_step = decay_step
        self.lower_bound = 0.0

    def step(self):
        self._step(self.alpha)
        self.lower_bound = min(self.lower_bound + self.decay_step, self.max_certanty)

    @property
    def alpha(self):
        return random.uniform(self.lower_bound, 1.0)


class SquareThresholdMSELoss:
    def __init__(self, threshold):
        self.threshold = threshold
        self.current_shape = None
        self.mask = None

    def get_mask(self, b, e, device):
        if self.current_shape != (b, e):
            self.current_shape = (b, e)
            self.mask = torch.zeros((b * e, b * e), device=device)
            for i in range(b):
                self.mask[i * e:(i + 1) * e, i * e:(i + 1) * e] = 1
        return self.mask

    def __call__(self, input):
        b, e, *_ = input.shape
        input = input.view(b * e, -1)
        input = torch.where(input < self.threshold, input, 1.0)
        mse_error = (input.unsqueeze(0) - input.unsqueeze(1)).pow(2).mean(-1)
        mask = self.get_mask(b, e, device=input.device)
        return (mse_error * mask).sum() / mask.sum()


class NoCudnnCTCLoss(torch.nn.CTCLoss):
    def __init__(self, *args, **kwargs):
        super(NoCudnnCTCLoss, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        torch.backends.cudnn.enabled = False
        loss = super(NoCudnnCTCLoss, self).forward(*args, **kwargs)
        torch.backends.cudnn.enabled = True
        return loss


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


class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_epochs, target_lr):
        """
        Create a linear warm-up learning rate scheduler as a class.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be adjusted.
            warmup_epochs (int): The number of epochs for the warm-up phase.
            target_lr (float): The target learning rate after warm-up.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr

        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                return (current_epoch + 1) / self.warmup_epochs * self.target_lr
            else:
                return self.target_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def step(self):
        """Update the learning rate for the current epoch."""
        self.scheduler.step()


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss_fake = -torch.mean(dis_fake)
    return loss_fake


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
