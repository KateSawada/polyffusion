# Copied from torch_plus
from typing import Tuple, List

import numpy as np


def scheduled_sampling(i, high=0.7, low=0.05):
    i /= 1000 * 40  # new update
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y


class _Scheduler:
    def __init__(self, step=0, mode="train"):
        self._step = step
        self._mode = mode

    def _update_step(self):
        if self._mode == "train":
            self._step += 1
        elif self._mode == "val":
            pass
        else:
            raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "val"


class ConstantScheduler(_Scheduler):
    def __init__(self, param, step=0.0):
        super(ConstantScheduler, self).__init__(step)
        self.param = param

    def step(self):
        self._update_step()
        return self.param


class TeacherForcingScheduler(_Scheduler):
    def __init__(self, high, low, f=scheduled_sampling, step=0):
        super(TeacherForcingScheduler, self).__init__(step)
        self.high = high
        self.low = low
        self._step = step
        self.schedule_f = f

    def get_tfr(self):
        return self.schedule_f(self._step, self.high, self.low)

    def step(self):
        tfr = self.get_tfr()
        self._update_step()
        return tfr


class MultiTeacherForcingScheduler(_Scheduler):
    def __init__(
            self,
            tf_rates: List[Tuple[float, float]],
            f=scheduled_sampling,
            step: int=0
    ):
        """TeacherForcingSchedulerを複数まとめたもの

        Args:
            tf_rates (List[Tuple[float, float]]): tf_rate. Each element in list
                holds (high, low)
            f (callable, optional): scheduled_sampling function
            step (int, optional): Initial step. Defaults to 0.
        """
        super(MultiTeacherForcingScheduler, self).__init__(step)
        self.schedulers = []
        self.tf_rates = tf_rates
        for i in range(len(tf_rates)):
            self.schedulers.append(
                TeacherForcingScheduler(*tf_rates[i], f, step))

    def step(self):
        tfrs = []
        for i in range(len(self.tf_rates)):
            tfrs.append(self.schedulers[i].step())
        return tfrs


class OptimizerScheduler(_Scheduler):
    def __init__(self, optimizer, scheduler, clip, step=0):
        # optimizer and scheduler are pytorch class
        super(OptimizerScheduler, self).__init__(step)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, require_zero_grad=False):
        self.optimizer.step()
        self.scheduler.step()
        if require_zero_grad:
            self.optimizer_zero_grad()
        self._update_step()


class ParameterScheduler(_Scheduler):
    def __init__(self, step=0, mode="train", **schedulers):
        # optimizer and scheduler are pytorch class
        super(ParameterScheduler, self).__init__(step)
        self.schedulers = schedulers
        self.mode = mode

    def train(self):
        self.mode = "train"
        for scheduler in self.schedulers.values():
            scheduler.train()

    def eval(self):
        self.mode = "val"
        for scheduler in self.schedulers.values():
            scheduler.eval()

    def step(self, require_zero_grad=False):
        params_dic = {}
        for key, scheduler in self.schedulers.items():
            params_dic[key] = scheduler.step()
        return params_dic
