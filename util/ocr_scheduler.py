import torch
import random
import math


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
