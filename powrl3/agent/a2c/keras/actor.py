#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Adapted from the following repos:
- https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
- https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl2/cartpole/pg_tf.py
- https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html
- https://github.com/arnomoonens/DeepRL
- https://github.com/keon/policy-gradient/blob/master/actor.py
- https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
- https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
- https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb
- https://github.com/jiexunsee/Deep-Watkins-Q-and-Actor-Critic/blob/master/PolicyLearnerCNN.py
"""

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Embedding, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras import backend as K

import numpy as np


class PolicyGradientModel(object):
    # TODO: use eligibility traces as it is explained here:
    # http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf

    def __init__(self, actions, n_dims, lr=1e-4, use_boltzmann_exploration=True):
        self.actions = actions
        self.n_actions = len(self.actions)
        self.n_dims = n_dims
        self.lr = lr

        self.use_boltzmann_exploration = use_boltzmann_exploration
        if use_boltzmann_exploration:
            self.t_boltzmann = 100.0
            self.n_steps = 1

        self.actions_predicted = []

        self._build_network(n_dims=self.n_dims, n_actions=self.n_actions)

        self.model.summary()

    def _build_network(self, n_dims, n_actions, hidden_dims=(32, 32)):
        """Create a base network"""
        self.X = Input(shape=(n_dims,), name="X")
        self.action_prob = Input(shape=(n_actions,), name="action_prob")
        self.returns = K.placeholder(shape=(None,), name="returns")

        x = self.X

        x = Embedding(256, output_dim=16)(x)  # suggestion: not greater than 16
        x = BatchNormalization()(x)  # TODO: is this necessary?

        x = Conv1D(filters=32, kernel_size=3, activation='relu',
                   padding='valid', strides=1)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu',
                   padding='valid', strides=1)(x)
        x = GlobalMaxPooling1D()(x)

        for h_dim in hidden_dims:
            x = Dense(h_dim, activation='relu')(x)
            # x = Dropout(0.2)(x)

        self.output = Dense(n_actions, activation='softmax', use_bias=False)(x)
        self.model = Model(inputs=self.X, outputs=self.output)

        # Loss function

        action_prob = K.sum(self.output * self.action_prob, axis=1)
        log_action_prob = K.log(action_prob)
        self.loss = -K.sum(log_action_prob * self.returns) #+ 0.01*K.mean(self.output*K.log(self.output))

        #self.optimizer = Adam(lr=self.lr)
        self.optimizer = Adadelta(rho=0.9)

        self.updates = self.optimizer.get_updates(params=self.model.trainable_weights, loss=self.loss)

        self.train_fn = K.function(inputs=[self.X,
                                           self.action_prob,
                                           self.returns],
                                   outputs=[],
                                   updates=self.updates)

    def train(self, x, a, r):
        self.train_fn([np.asarray(x, dtype=np.float32),
                       np.asarray(a, dtype=np.float32),
                       np.asarray(r, dtype=np.float32)])

    def predict(self, x):
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x).flatten()

    def update_t_boltzmann(self):
        if self.use_boltzmann_exploration:
            self.t_boltzmann /= np.sqrt(self.n_steps)
            self.n_steps += 1

    def reset_t_boltzmann(self):
        if self.use_boltzmann_exploration:
            self.t_boltzmann = 100.0
            self.n_steps = 1

    def get_boltzmann_values(self, x):
        p = self.predict(x)
        max_p = max(p)
        exp_p = [np.exp((p_i - max_p)/float(self.t_boltzmann)) for p_i in p]
        sum_exp_p = sum(exp_p)
        softmax_p = [exp_p_i / float(sum_exp_p) for exp_p_i in exp_p]
        return softmax_p

    def sample_action(self, x, explore=False):
        actions = sorted(self.actions.items(), key=lambda t: t[1])
        if explore:
            # Explore through Boltzmann method
            # probability_actions = self.get_boltzmann_values(s)
            probability_actions = self.predict(x)
            a = np.random.choice([t[0] for t in actions], p=probability_actions)
        else:
            # Exploit
            probability_actions = self.predict(x)
            a_i, p = max(enumerate(probability_actions), key=lambda t: t[1])
            a, i = filter(lambda t: t[1] == a_i, actions)[0]

        # print("Probability actions: %s" % str(probability_actions))

        return a

    def load(self, filename):
        self.model.load_weights(filename)
        return self

    def save(self, filename):
        try:
            self.model.save_weights(filename)
            success = True
        except:
            success = False

        return success
