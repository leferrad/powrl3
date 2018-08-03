#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Example of solving a magic cube with Actor-Critic model"""

from __future__ import print_function

__author__ = 'leferrad'

from powrl3.agent.feature import LBPFeatureTransformer
from powrl3.environment.cube import CubeEnvironment
from powrl3.environment.curriculum import max_movements, get_max_movements_probabilistic
from powrl3.util.fileio import get_logger
from powrl3.util.plot import plot_moving_avg, plot

import powrl3.agent.a2c.keras.base as a2ck
import powrl3.agent.a2c.mxnet.base as a2cmx

import numpy as np
import random

import argparse


logger = get_logger(name="main", level="debug")


backends = {"keras": a2ck, "mxnet": a2cmx}

# NOTE: It is important to generate a seed int only through 'random' module, since it will feed the 'np.random' module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of solving a magic cube with Actor-Critic model")
    parser.add_argument('-g', '--gamma', type=float, dest='gamma', default=0.99, help='Discount factor')
    parser.add_argument('-s', '--seed', type=int, dest='seed', default=123, help='Random seed')
    parser.add_argument('-ne', '--n-epis', type=int, dest='N', default=5000, help='Number of episodes to run')
    parser.add_argument('-cm', '--cube-movs', type=int, dest='M', default=5, help='Total of movements for the cube')
    parser.add_argument('-rm', '--rep-mem', dest='replay_mem', type=int, default=50000,
                        help='Max amount of sequences to store in experience replay memory')
    parser.add_argument('-sr', '--steps-replay', type=int, dest='sr', default=50,
                        help='Number of steps to wait to apply experience replay')
    parser.add_argument('-rf', '--reward-function', dest='reward_function', default="standard",
                        help='Reward function to use (either "simple" or "standard")')
    parser.add_argument('-o', '--out-model', dest='out_model', default='/tmp/powerl3_model.tgz',
                        help='Output path to store resulting model')
    parser.add_argument('-b', '--backend', dest='backend', default='mxnet',
                        help='Backend to use (either "keras" or "mxnet")')

    args = parser.parse_args()

    print("Using seed=%i" % args.seed)

    random.seed(args.seed)
    seed = random.randint(0, 1000)

    ce = CubeEnvironment(seed=seed, reward_function=args.reward_function)
    ce.randomize(1)

    path_to_model = args.out_model

    backend = args.backend

    if backend not in backends:
        logger.warning("Backend '%s' not supported! Changing to default backend: '%s'",
                       str(backend), "tf")
        backend = "tf"

    a2c = backends[backend]

    # Create actor-critic models
    N = args.N
    M = args.M
    lr = 1e-4

    ac_agent = (a2c.ACAgent(env=ce, feature_transformer=LBPFeatureTransformer(), gamma=args.gamma,
                            max_memory=args.replay_mem, steps_replay=args.sr, lr=lr)
                  #.load(path_to_model)
                )

    max_movements_for_cur_iter, max_movements_for_random = 0, 0
    total_rewards = np.empty(N)
    total_iters = np.empty(N)
    total_games = np.empty(N)

    for n in range(N):
        prev_max_movements_for_cur_iter, prev_max_movements_for_random = max_movements_for_cur_iter, max_movements_for_random
        max_movements_for_cur_iter, max_movements_for_random = max_movements(n, N, M)
        if prev_max_movements_for_cur_iter != max_movements_for_cur_iter:
            logger.info("Now playing for a max of %i movements..." % max_movements_for_cur_iter)
        total_reward, iters, solved = a2c.play_one(ac_agent, ce, max_iters=max_movements_for_cur_iter)
        total_rewards[n] = total_reward
        total_iters[n] = iters
        total_games[n] = int(solved)

        if n % 100 == 0:
            logger.info("Episode:%i, total reward: %s, avg reward (last 100): %s, avg games solved (last 100): %s",
                        n, total_reward, total_rewards[max(0, n-100):(n+1)].mean(),
                        total_games[max(0, n-100):(n+1)].mean())
            #print(ce.render())

        seed = random.randint(0, 1000)
        ce.reset()
        if prev_max_movements_for_random != max_movements_for_random:
            logger.info("Now randomizing environment with a max of %i movements..." % max_movements_for_random)

        ce.randomize(get_max_movements_probabilistic(max_movements_for_random))

    logger.info("avg reward for last 100 episodes: %s", total_rewards[-100:].mean())
    logger.info("total steps: %s", total_rewards.sum())

    plot(y=total_rewards, title="Rewards", show=True)

    plot_moving_avg(total_rewards, title="Total rewards per episode- moving average")
    plot_moving_avg(total_iters, title="Total iterations per episode - moving average")
    plot_moving_avg(total_games, title="Total games solved - moving average")

    logger.info("Saving model on %s..." % path_to_model)
    ac_agent.save(path_to_model)

    logger.info("Done!")
