#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""""""

# Links:
# http://mi.eng.cam.ac.uk/~mg436/LectureSlides/MLSALT7/L5.pdf
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2

from powrl3.agent.a2c.mxnet.actor import PolicyGradientModel
from powrl3.agent.a2c.mxnet.critic import ValueEstimator
from powrl3.agent.a2c.util import ExperienceReplayAC
from powrl3.util.fileio import compress_tar_files, decompress_tar_files

from keras import utils as np_utils

import os


class ACAgent(object):
    def __init__(self, env, feature_transformer,
                 gamma=0.99, lr=1e-4,
                 max_memory=10000, steps_replay=50):
        self.env = env
        self.actions = dict([(a, i) for (i, a) in enumerate(env.actions_available.keys())])
        self.n_actions = len(self.actions)
        self.n_dims = feature_transformer.dimensions
        self.feature_transformer = feature_transformer
        self.gamma = gamma
        self.memory = ExperienceReplayAC(max_items=max_memory)

        # Actor
        self.policy_estimator = PolicyGradientModel(actions=self.actions, n_dims=self.n_dims, lr=lr)
        # Critic
        #self.value_estimator = ValueEstimator(n_hidden=200)
        self.value_estimator = ValueEstimator(n_dims=self.n_dims)

        self.steps_replay = steps_replay
        self.steps = 0

    def update(self, s, a, r, s_prime, t):
        # Add this tuple to the current memory
        s_x = self.feature_transformer.transform(s, normalize=False)
        s_prime_x = self.feature_transformer.transform(s_prime, normalize=False)
        self.memory.add(s=s_x, a=a, r=r, s_prime=s_prime_x, t=t)

        if False: #self.steps > self.steps_replay:
            # After a max of steps, it's time to apply experience replay

            # Get a batch of data from the memory
            data = self.memory.get_sample(max_items=200)

            states_x, actions, probs, values, advantages = [], [], [], [], []

            for state_x, action, reward, next_state_x, terminal in data:

                # TODO: save both action and value in ER memory?

                # action: map str to int
                action = self.actions[action]

                # Calculate values for Actor and Critic
                next_value = self.value_estimator.predict(next_state_x)[0]

                z = self.policy_estimator.predict(state_x)

                if terminal:
                    value = reward
                else:
                    value = reward + self.gamma * next_value

                advantage = value - self.value_estimator.predict(state_x)[0]

                states_x.append(state_x)
                actions.append(action)
                values.append(value)
                advantages.append(advantage)
                probs.append(z)

            self.steps = 0  # reset the counter

            # Update both Actor and Critic
            self.policy_estimator.train(x=states_x, a=probs, r=advantages)
            self.value_estimator.train(x=states_x, v=advantages)

        else:
            # Otherwise, just fit the current input

            # Calculate values for Actor and Critic
            next_value = self.value_estimator.predict(s_prime_x)[0]

            if t:
                value = r
            else:
                value = r + self.gamma * next_value

            advantage = value - self.value_estimator.predict(s_x)[0]

            z = np_utils.to_categorical(self.actions[a], num_classes=self.n_actions)

            states_x, probs, values, advantages = [s_x], [z], [value], [advantage]

            # Update both Actor and Critic
            # TODO: value function learns advantages or values?
            # See http://www.ausy.tu-darmstadt.de/uploads/Teaching/RobotLearningLecture2015/Value_Function_Methods.pdf
            self.policy_estimator.train(x=states_x, a=probs, r=advantages)
            self.value_estimator.train(x=states_x, v=advantages)

        self.steps += 1

    def sample_action(self, s):
        x = self.feature_transformer.transform(s, normalize=False)
        return self.policy_estimator.sample_action(x, explore=True)

    def save(self, filename):
        base_path = os.path.dirname(filename)

        paths = []

        # 1) Save the model of Value Estimator
        value_estimator_path = os.path.join(base_path, 'value_estimator.tgz')
        success = self.value_estimator.save(value_estimator_path)
        paths.append(value_estimator_path)

        if success:
            # 2) Save the model of Policy Estimator
            policy_estimator_path = os.path.join(base_path, 'policy_estimator.tgz')
            success = self.policy_estimator.save(policy_estimator_path)
            paths.append(policy_estimator_path)

        if success:
            # 3) Compress all the files
            success = compress_tar_files(files=paths, filename=filename)

        try:
            # remove temporary files which are already compressed
            for p in paths:
                os.remove(p)
        except:
            pass

        return success

    def load(self, filename):
        success = decompress_tar_files(filename)
        if success:
            base_path = os.path.dirname(filename)

            for path in os.listdir(base_path):
                if path.startswith('policy_'):
                    # Then it corresponds to a Policy Estimator
                    path = os.path.join(base_path, path)
                    self.policy_estimator = (PolicyGradientModel(actions=self.actions,
                                                                 n_dims=self.n_dims
                                                                 ).load(path))
                    try:
                        # remove temporary files which are already decompressed
                        os.remove(path)
                    except:
                        pass
                elif path.startswith('value_'):
                    # Then it corresponds to a Value Estimator
                    path = os.path.join(base_path, path)
                    self.value_estimator = (ValueEstimator(n_dims=self.n_dims).load(path))
                    try:
                        # remove temporary files which are already decompressed
                        os.remove(path)
                    except:
                        pass

        return self


def play_one(model, env, max_iters=100):
    env.actions_taken = []  # Reset actions taken on the scramble stage
    observation = env.get_state()

    total_reward = 0
    iters = 0

    """
    try:
        model.reset()
    except:
        pass
    """

    while not env.is_solved() and iters < max_iters:
        # Take a step
        action = model.sample_action(observation)
        next_observation, reward, solved = env.take_action(action)
        total_reward += reward

        model.update(s=observation, a=action, r=reward, s_prime=next_observation, t=solved)

        if env.is_solved():
            print("WOW! The cube is solved! Algorithm followed: %s" % str(env.actions_taken))

        observation = next_observation
        iters += 1

    # TODO: return total value?

    return total_reward, iters, env.is_solved()

