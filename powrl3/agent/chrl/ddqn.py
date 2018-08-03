#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""""""

# https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/

from powrl3.agent.chrl.base import BaseAgent

from chainer import links as L
from chainerrl import links
from chainerrl.agents import DoubleDQN
from chainerrl.q_functions import FCStateQFunctionWithDiscreteAction

import chainerrl


import numpy as np


def phi(obs):
    return obs.astype(np.float32, copy=False)


def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))


class DoubleDQNAgent(BaseAgent):
    def __init__(self, env, feature_transformer, gamma=0.99, optimizer='adam', max_memory=10000):
        BaseAgent.__init__(self, env=env, feature_transformer=feature_transformer, gamma=gamma,
                           optimizer=optimizer)

        self.model = links.Sequence(L.ConvolutionND(ndim=1, in_channels=self.n_dims,
                                                    out_channels=100, ksize=3,
                                                    stride=1, pad=1, cover_all=True),
                                    FCStateQFunctionWithDiscreteAction(ndim_obs=100, n_actions=self.n_actions,
                                                                       n_hidden_channels=100, n_hidden_layers=2)
                                    )

        self.optimizer.setup(self.model)
        #self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))

        self.replay_buffer = \
            chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(
                capacity=max_memory,
                uniform_ratio=0.1,
                default_priority_func=exp_return_of_episode,
                wait_priority_after_sampling=False,
                return_sample_weights=False)

        self.agent = DoubleDQN(q_function=self.model, optimizer=self.optimizer, replay_buffer=self.replay_buffer,
                               explorer=self.explorer, gamma=self.gamma, phi=phi,
                               update_interval=500, minibatch_size=50)

