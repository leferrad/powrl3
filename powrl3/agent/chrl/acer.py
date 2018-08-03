#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from powrl3.agent.chrl.base import BaseAgent

from chainer import links as L
from chainer import functions as F
from chainerrl.action_value import DiscreteActionValue
from chainerrl.agents import acer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl.replay_buffer import PrioritizedEpisodicReplayBuffer  # TODO: make a module to override this class
from chainerrl import links

import numpy as np


# http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html


def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))


class ACERAgent(BaseAgent):
    def __init__(self, env, feature_transformer, gamma=0.99, optimizer='adam', max_memory=10000):
        BaseAgent.__init__(self, env=env, feature_transformer=feature_transformer, gamma=gamma,
                           optimizer=optimizer)

        self.model = acer.ACERSharedModel(shared=links.Sequence(L.ConvolutionND(ndim=1, in_channels=self.n_dims,
                                                                                out_channels=100, ksize=3,
                                                                                stride=1, pad=1, cover_all=True),
                                                                L.Linear(100, 100),
                                                                F.relu),
                                          pi=links.Sequence(L.Linear(100, self.n_actions),
                                                            F.relu,
                                                            SoftmaxDistribution),
                                          q=links.Sequence(L.Linear(100, self.n_actions),
                                                           F.relu,
                                                           DiscreteActionValue))

        self.optimizer.setup(self.model)
        #self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))

        self.replay_buffer = PrioritizedEpisodicReplayBuffer(capacity=max_memory,
                                                             uniform_ratio=0.1,
                                                             default_priority_func=exp_return_of_episode,
                                                             wait_priority_after_sampling=False,
                                                             return_sample_weights=False)

        self.agent = acer.ACER(model=self.model, optimizer=self.optimizer, gamma=self.gamma,
                               replay_buffer=self.replay_buffer, phi=self.phi,
                               n_times_replay=2, t_max=200, replay_start_size=50,
                               disable_online_update=False, use_trust_region=False, use_Q_opc=True,
                               trust_region_delta=0.1, truncation_threshold=None, beta=1e-2)
