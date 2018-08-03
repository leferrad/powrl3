#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapted from the following repos:
- https://github.com/keon/policy-gradient/blob/master/actor.py
- https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
- https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
- https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb
"""

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Embedding, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta

import numpy as np


class ValueEstimator(object):
    # TODO: use eligibility traces as it is explained here:
    # http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf

    def __init__(self, n_dims, lr=1e-4):
        self.n_dims = n_dims
        self.lr = lr

        self._build_network(n_dims=self.n_dims)

        self.model.summary()  # print_fn=logger

    """
    def build_network2(self, n_dims, n_actions, hidden_dims=(64, 64)):
        # TODO: exp moving avg ?
        x = Input(shape=(n_dims,))
        a = Input(shape=(n_actions,))

        net, net_a = x, a

        net = Embedding(256, output_dim=16)(net)  # suggestion: not greater than 16
        net = BatchNormalization()(net)  # TODO: is this necessary?

        net = Conv1D(filters=32, kernel_size=3, activation='relu',
                     padding='valid', strides=1)(net)
        net = GlobalMaxPooling1D()(net)

        for h_dim in hidden_dims:
            net = Dense(h_dim, activation='relu')(net)
            net_a = Dense(h_dim, activation='relu')(net_a)

        z = add([net, net_a])
        #z = Dense(32, activation='relu')(z)
        z = Dense(1, activation='linear')(z)

        model = Model(inputs=[x, a], outputs=z)
        #opt = Adam(lr=self.lr)
        opt = Adadelta(rho=0.99)
        model.compile(loss='mse', optimizer=opt)

        return x, a, model
    """

    def _build_network(self, n_dims, hidden_dims=(32, 32)):
        # TODO: exp moving avg ?
        self.X = Input(shape=(n_dims,))

        net = self.X

        net = Embedding(256, output_dim=16)(net)  # suggestion: not greater than 16
        net = BatchNormalization()(net)  # TODO: is this necessary?

        net = Conv1D(filters=32, kernel_size=3, activation='relu',
                     padding='valid', strides=1)(net)
        net = GlobalMaxPooling1D()(net)

        for h_dim in hidden_dims:
            net = Dense(h_dim, activation='relu')(net)

        self.output = Dense(1, activation='linear')(net)

        self.model = Model(inputs=self.X, outputs=self.output)

        self.optimizer = Adadelta(rho=0.9)
        #self.optimizer = Adam(lr=self.lr)

        self.model.compile(loss='mse', optimizer=self.optimizer)

    def train(self, x, v):
        self.model.train_on_batch(x=np.asarray(x, np.float32), y=np.asarray(v))

    def predict(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = np.asarray(x)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        return self.model.predict(x).flatten()

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