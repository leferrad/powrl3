#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""""""

from powrl3.agent.chrl.base import BaseAgent

from chainerrl import agents
from chainerrl import policies
from chainerrl import v_functions
import chainerrl

import numpy as np


def phi(obs):
    return obs.astype(np.float32, copy=False)


def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))


class PCLAgent(BaseAgent):
    def __init__(self, env, feature_transformer, gamma=0.99, optimizer='adam', max_memory=10000):
        BaseAgent.__init__(self, env=env, feature_transformer=feature_transformer, gamma=gamma,
                           optimizer=optimizer)

        self.model = agents.pcl.PCLSeparateModel(
            pi=policies.FCSoftmaxPolicy(
                self.n_dims, self.n_actions,
                n_hidden_channels=100,
                n_hidden_layers=2
            ),
            v=v_functions.FCVFunction(
                self.n_dims,
                n_hidden_channels=100,
                n_hidden_layers=2,
            ),
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

        self.agent = agents.pcl.PCL(
                    model=self.model, optimizer=self.optimizer,
                    replay_buffer=self.replay_buffer,
                    t_max=1, gamma=self.gamma,
                    tau=1e-2,
                    phi=phi, rollout_len=10, batchsize=1,
                    disable_online_update=False, n_times_replay=1,
                    replay_start_size=1000, normalize_loss_by_steps=True,
                    act_deterministically=False,
                    backprop_future_values=False,
                    train_async=True
                )



