#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""""""

from powrl3.util.fileio import get_logger

from chainer import optimizers
from chainerrl import explorers
from chainerrl.experiments.evaluator import save_agent

import numpy as np


logger = get_logger(name=__name__)


# Choices that make most sense for this environment, and using most stable settings
available_optimizers = {'rms': optimizers.RMSprop(lr=1e-4),
                        'adam': optimizers.Adam(alpha=1e-3),
                        'adadelta': optimizers.AdaDelta(rho=0.95)}

available_explorers = {'boltzmann': explorers.Boltzmann(T=1.0),
                       'lin_decay_eps_greedy': explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                                                                  end_epsilon=0.0,
                                                                                  decay_steps=10,
                                                                                  random_action_func=None,
                                                                                  logger=None),
                       'const_eps_greedy': explorers.ConstantEpsilonGreedy(epsilon=1.0, random_action_func=None,
                                                                           logger=None)}


class BaseAgent(object):
    def __init__(self, env, feature_transformer, gamma=0.99, optimizer='adadelta', explorer='boltzmann'):
        self.actions = dict([(a, i) for (i, a) in enumerate(env.actions_available)])
        self.n_actions = len(self.actions)
        self.n_dims = feature_transformer.dimensions
        self.feature_transformer = feature_transformer
        self.gamma = gamma
        self.optimizer_algorithm = optimizer
        self.optimizer = available_optimizers[self.optimizer_algorithm]
        self.explorer = available_explorers[explorer]
        self.agent = None

    @staticmethod
    def phi(obs):
        return obs.astype(np.float32, copy=False)

    def act_and_train(self, state, reward):
        if self.agent is None:
            raise ValueError("Agent was not initialized!")

        state_x = self.feature_transformer.transform(state, normalize=True)
        state_x = np.atleast_2d(state_x)
        state_x = np.resize(state_x, (self.n_dims, 1))

        return self.agent.act_and_train(state_x, reward)

    def stop_episode_and_train(self, state, reward, done):
        state_x = self.feature_transformer.transform(state, normalize=True)
        state_x = np.atleast_2d(state_x)
        state_x = np.resize(state_x, (self.n_dims, 1))

        return self.agent.stop_episode_and_train(state_x, reward, done)

    def action_int_to_str(self, a_int):
        res = list(filter(lambda t: t[1] == a_int, self.actions.items()))

        if len(res) == 0:
            return None

        # Return the only result, just the key wanted
        return res[0][0]

    def get_statistics(self):
        return self.agent.get_statistics()

    def save(self, dirname):
        self.agent.save(dirname)

    def load(self, dirname):
        self.agent.load(dirname)


def play_one(agent, env, max_steps=100):
    observation = env.get_state()

    total_reward = 0
    steps = 0
    reward = 0

    while not env.is_solved() and steps < max_steps:
        # Take a step
        action = agent.act_and_train(observation, reward)

        action = agent.action_int_to_str(action)

        next_observation, reward, solved = env.take_action(action)
        total_reward += reward

        if solved:
            logger.info("WOW! The cube is solved! Algorithm followed: %s" % str(env.actions_taken))
            agent.stop_episode_and_train(observation, reward, done=solved)

        observation = next_observation
        steps += 1

    return total_reward, steps, env.is_solved()

