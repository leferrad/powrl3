#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model of the environment that will be used for every agent"""

__author__ = 'leferrad'

from pycuber import Cube
import numpy as np
import copy


def simple_reward(cube, reward_positive=10, reward_negative=-0.1):
    reward = reward_negative
    if cube.is_solved():
        reward = reward_positive
    return reward


def are_inverse_actions(a1, a2):
    are_inverse = False
    if a1 != a2 and any([a1.replace("'", "") == a2, a2.replace("'", "") == a1]):
        are_inverse = True
    return are_inverse


# See https://hal.archives-ouvertes.fr/hal-00331752/document
# the shortest way toward the goal is the sought optimal
# policy concerning goal-directed tasks. So rg must always be superior to Q∞.

def standard_reward(cube, reward_positive=10, reward_negative=-0.1):
    if cube.is_solved():
        reward = reward_positive
    else:
        reward = 0.0
    actions_taken = cube.actions_taken
    # punish to avoid regrets (i.e. agent moving back and forth between two states)
    if len(actions_taken) >= 2 and are_inverse_actions(actions_taken[-2], actions_taken[-1]):
        reward += 10 * reward_negative
    # punish to avoid loops (i.e. agent doing a complete loop up to the same state,
    #                        or making three steps that are equal to make just a single inverse step)
    elif len(actions_taken) >= 3 and len(set(actions_taken[-3:])) == 1:
        reward += 10 * reward_negative
    else:
        reward += reward_negative  # due to take a step
    return reward


rewards_available = {'simple': simple_reward, 'standard': standard_reward}


class CubeEnvironment(object):
    faces = {"U", "D", "F", "B", "R", "L"}
    colour_dict = {"red": 0, "yellow": 1, "green": 2, "white": 3, "orange": 4, "blue": 5}
    actions_available = {"U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"}
    n_actions = len(actions_available)

    def __init__(self, reward_function='simple', seed=123):
        self.cube = Cube()
        self.actions_taken = []
        self.reward_function = rewards_available[reward_function]

        self.seed = seed
        np.random.seed(seed)

    def is_solved(self):
        return all([len(set(np.asarray(self.cube.get_face(face)).flatten())) == 1 for face in self.faces])

    def move(self, step):
        self.cube.perform_step(step)

    def get_state(self):
        state = []
        # Sorted list to always return the same order
        for face in sorted(self.faces):
            for row in self.cube.get_face(face):
                for square in row:
                    state.append(self.colour_dict[square.colour])

        return state

    def take_action(self, action):
        assert action in self.actions_available, \
            ValueError("Action '%s' doesn't belong to the supported actions: %s",
                       str(action), str(self.actions_available))
        self.move(action)
        self.actions_taken.append(action)
        reward = self.reward_function(self)
        return self.get_state(), reward, self.is_solved()

    @staticmethod
    def sample_action():
        return np.random.choice(list(CubeEnvironment.actions_available))

    def render(self):
        return self.cube.__repr__()

    def randomize(self, n=20):
        for _ in range(n):
            action = self.sample_action()
            self.move(action)  # do not track this movement in 'actions_taken'

    def copy(self, deep=True):
        return copy.deepcopy(self) if deep else copy.copy(self)

    def reset(self):
        self.cube = Cube()
        self.actions_taken = []
