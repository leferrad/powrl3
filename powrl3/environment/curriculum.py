#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Curriculum learning functions to achieve a progressive learning of the agent"""

__author__ = 'leferrad'


import numpy as np


def max_movements(n_iter, max_iters, total_movements=10, clip_movements=50):
    max_movs_for_random = int((n_iter * total_movements) / int(max_iters) + 1)
    # M = max_iters / total_movements
    # max_movs = int(n_iter / M + 1) * total_movements
    # max_movs = max_movs_for_random ** 3
    max_movs = int(2.5 ** max_movs_for_random)
    # max_movs = int(2.4270509831248424 ** max_movs_for_random)  # num_aureo * n_caras / n vertices
    max_movs = min(max_movs, clip_movements)

    return max_movs, max_movs_for_random


def get_max_movements_probabilistic(n_movements):
    movements = range(1, int(n_movements + 1))
    sum_movements = sum(movements)
    probabilities = [m/float(sum_movements) for m in movements]
    return int(np.random.choice(movements, p=probabilities))

