#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Example of solving a magic cube with models built on chainerrl"""

__author__ = 'leferrad'


from powrl3.agent.feature import LBPFeatureTransformer
from powrl3.environment.cube import CubeEnvironment
from powrl3.environment.curriculum import max_movements, get_max_movements_probabilistic
from powrl3.util.fileio import get_logger
from powrl3.util.plot import plot_moving_avg, plot

from powrl3.agent.chrl import base, pcl, ppo, acer, reinforce, ddqn

import numpy as np
import random

import argparse

# NOTE: It is important to generate a seed only through 'random' module, since it will feed the 'np.random' module

logger = get_logger(name=__name__)

agent_models = {"pcl": pcl.PCLAgent, "ppo": ppo.PPOAgent, "acer": acer.ACERAgent,
                "reinforce": reinforce.REINFORCEAgent, "ddqn": ddqn.DoubleDQNAgent}


def train_agent(agent, env, n_episodes=5000, total_movements=5,
                step_offset=0, outdir='/tmp/results', step_hooks=None):

    max_movements_for_cur_iter, max_movements_for_random = 0, 0
    total_rewards = np.empty(n_episodes)
    total_iters = np.empty(n_episodes)
    total_games = np.empty(n_episodes)

    n = step_offset

    try:
        while n < n_episodes:

            prev_max_movements_for_cur_iter = max_movements_for_cur_iter
            prev_max_movements_for_random = max_movements_for_random
            max_episode_len, max_movements_for_random = max_movements(n, n_episodes, total_movements)
            if prev_max_movements_for_cur_iter != max_movements_for_cur_iter:
                logger.info("Now playing for a max of %i movements..." % max_movements_for_cur_iter)

            episode_r, episode_len, solved = base.play_one(agent, env, max_steps=max_episode_len)

            total_rewards[n-step_offset] = episode_r
            total_iters[n-step_offset] = episode_len
            total_games[n-step_offset] = int(solved)

            n += 1

            if n % 100 == 0:
                logger.info("Episode: %i, steps done: %i, episode reward: %s, "
                            "avg reward (last 100): %s, avg games solved (last 100): %s",
                            n, episode_len, episode_r, total_rewards[max(0, n - 100):(n + 1)].mean(),
                            total_games[max(0, n - 100):(n + 1)].mean())
                logger.info('Statistics:%s', agent.get_statistics())
                #print(ce.render())

            # seed = random.randint(0, 1000)
            env.reset()
            if prev_max_movements_for_random != max_movements_for_random:
                logger.info("Now randomizing environment with a max of %i movements..." % max_movements_for_random)

            env.randomize(get_max_movements_probabilistic(max_movements_for_random))

            while env.is_solved():
                env.randomize(get_max_movements_probabilistic(max_movements_for_random))

            if isinstance(step_hooks, list):
                for hook in step_hooks:
                    hook(env, agent, n)

            if n == n_episodes:
                break

    # except KeyboardInterrupt
    except Exception:
        # Save the current model before being killed
        logger.info("Saving model on %s..." % outdir)
        base.save_agent(agent, n, outdir, logger, suffix='_except')
        raise

    # Save the final model
    logger.info("Saving model on %s..." % outdir)
    base.save_agent(agent, n, outdir, logger, suffix='_finish')

    logger.info("avg reward for last 100 episodes: %s", str(total_rewards[-100:].mean()))
    logger.info("total steps: %s", str(total_rewards.sum()))

    plot(y=total_rewards, title="Rewards", show=True)

    plot_moving_avg(total_rewards, title="Total rewards per episode- moving average")
    plot_moving_avg(total_iters, title="Total iterations per episode - moving average")
    plot_moving_avg(total_games, title="Total games solved - moving average")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of solving a magic cube with agents built in ChainerRL")
    parser.add_argument('-g', '--gamma', type=float, dest='gamma', default=0.99, help='Discount factor')
    parser.add_argument('-s', '--seed', type=int, dest='seed', default=123, help='Random seed')
    parser.add_argument('-ne', '--n-epis', type=int, dest='N', default=4000, help='Number of episodes to run')
    parser.add_argument('-cm', '--cube-movs', type=int, dest='M', default=8, help='Total of movements for the cube')
    parser.add_argument('-rm', '--rep-mem', dest='replay_mem', type=int, default=10000,
                        help='Max amount of sequences to store in experience replay memory')
    parser.add_argument('-sr', '--steps-replay', type=int, dest='sr', default=100,
                        help='Number of steps to wait to apply experience replay')
    parser.add_argument('-rf', '--reward-function', dest='reward_function', default="simple",
                        help='Reward function to use (either "simple" or "standard")')
    parser.add_argument('-o', '--out-model', dest='out_model', default='/tmp/powerl3_model.tgz',
                        help='Output path to store resulting model')
    parser.add_argument('-m', '--model', dest='model', default='ppo',
                        help='Model to use (either "ppo", "pcl, "acer", "reinforce" or "ddqn")')

    args = parser.parse_args()

    logger.info("Using seed=%i" % args.seed)
    logger.info("Using '%s' reward function for environment" % args.reward_function)

    random.seed(args.seed)
    seed = random.randint(0, 1000)

    ce = CubeEnvironment(reward_function=args.reward_function, seed=seed)
    ce.randomize(1)

    path_to_model = args.out_model

    model = args.model

    if model not in agent_models:
        logger.warning("Model '%s' not supported! Changing to default model: '%s'",
                       str(model), "ppo")
        model = "ppo"

    agent_model = agent_models[model]

    agent = agent_model(env=ce, feature_transformer=LBPFeatureTransformer(),
                        gamma=args.gamma, max_memory=args.replay_mem)

    train_agent(agent=agent, env=ce, n_episodes=args.N, total_movements=args.M, outdir=path_to_model)

