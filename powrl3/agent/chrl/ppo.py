#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""""""

# https://medium.com/@sanketgujar95/trust-region-policy-optimization-trpo-and-proximal-policy-optimization-ppo-e6e7075f39ed

from powrl3.agent.chrl.base import BaseAgent

from chainer import links as L
from chainer import functions as F
from chainerrl import agents, links, policies, v_functions
from chainerrl.agents.a3c import A3CModel

import chainer

import numpy as np


def phi(obs):
    return obs.astype(np.float32, copy=False)


def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))


class A3CFF(chainer.ChainList, A3CModel):

    def __init__(self, n_dims, n_actions):
        self.head = links.Sequence(L.ConvolutionND(ndim=1, in_channels=n_dims,
                                                   out_channels=100, ksize=3,
                                                   stride=1, pad=1, cover_all=True),
                                   F.relu)
        self.pi = policies.FCSoftmaxPolicy(n_input_channels=100, n_actions=n_actions,
                                           n_hidden_layers=2, n_hidden_channels=100)
        self.v = v_functions.FCVFunction(n_input_channels=100, n_hidden_layers=2, n_hidden_channels=100)

        super(A3CFF, self).__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


class PPOAgent(BaseAgent):
    def __init__(self, env, feature_transformer, gamma=0.99, optimizer='adam', max_memory=100):
        BaseAgent.__init__(self, env=env, feature_transformer=feature_transformer, gamma=gamma,
                           optimizer=optimizer)

        self.model = A3CFF(n_dims=self.n_dims, n_actions=self.n_actions)

        self.optimizer.setup(self.model)
        #self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))

        self.agent = agents.ppo.PPO(
                    model=self.model, optimizer=self.optimizer,
                    gamma=self.gamma, phi=phi, lambd=0.7,
                    update_interval=500, minibatch_size=50, epochs=5,
                    standardize_advantages=True,
                )


