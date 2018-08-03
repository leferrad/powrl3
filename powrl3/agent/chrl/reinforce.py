#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from powrl3.agent.chrl.base import BaseAgent

from chainer import functions as F
from chainerrl.agents import reinforce
from chainerrl.replay_buffer import PrioritizedEpisodicReplayBuffer  # TODO: make a module to override this class
from chainerrl import policies

import numpy as np


def phi(obs):
    return obs.astype(np.float32, copy=False)


def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))


class REINFORCEAgent(BaseAgent):
    def __init__(self, env, feature_transformer, gamma=0.99, optimizer='adam', max_memory=10000):
        BaseAgent.__init__(self, env=env, feature_transformer=feature_transformer, gamma=gamma,
                           optimizer=optimizer)

        self.model = policies.FCSoftmaxPolicy(self.n_dims, self.n_actions,
                                              n_hidden_layers=2, n_hidden_channels=100,
                                              nonlinearity=F.relu)

        self.optimizer.setup(self.model)
        #self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))

        self.replay_buffer = PrioritizedEpisodicReplayBuffer(capacity=max_memory,
                                                             uniform_ratio=0.1,
                                                             default_priority_func=exp_return_of_episode,
                                                             wait_priority_after_sampling=False,
                                                             return_sample_weights=False)
        self.agent = reinforce.REINFORCE(model=self.model, optimizer=self.optimizer, phi=phi,
                                         batchsize=1, act_deterministically=False)
