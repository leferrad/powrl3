#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapted from the following repos:
- https://github.com/keon/policy-gradient/blob/master/actor.py
- https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
- https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
- https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb
"""

import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon

from mxnet import nd, autograd
from mxnet.gluon import nn

# TODO: use minpy?
import numpy as np


class Net(gluon.Block):
    def __init__(self, n_dims, hidden_dims=(32, 32), **kwargs):
        super(Net, self).__init__(**kwargs)

        self.n_dims = n_dims
        self.hidden_dims = hidden_dims

        with self.name_scope():

            self.embedding = nn.Embedding(256, output_dim=32)  # suggestion: not greater than 16
            self.bn = nn.BatchNorm()  # TODO: is this necessary?
            self.conv1 = nn.Conv1D(channels=32, kernel_size=3, activation='relu',
                                   padding=0, strides=1)
            #self.conv2 = nn.Conv1D(channels=32, kernel_size=3, activation='relu',
            #                       padding=0, strides=1)

            self.pool = nn.GlobalMaxPool1D()

            self.h1 = nn.Dense(32, activation='relu')
            #self.h2 = nn.Dense(32, activation='relu')

            self.output = nn.Dense(1)

    def forward(self, x):
        x = self.embedding(nd.array(x))
        x = self.bn(x)

        x = self.pool(self.conv1(x))

        x = self.h1(x)

        return F.identity(self.output(x))


class ValueEstimator(object):
    # TODO: use eligibility traces as it is explained here:
    # http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf

    def __init__(self, n_dims, lr=1e-4, ctx=mx.cpu(0)):
        self.n_dims = n_dims
        self.lr = lr
        self.ctx = ctx

        self.model, self.trainer = self._build_network(n_dims=self.n_dims, ctx=self.ctx)

    @staticmethod
    def _build_network(n_dims, hidden_dims=(32, 32), ctx=mx.cpu(0)):
        model = Net(n_dims=n_dims, hidden_dims=hidden_dims)
        model.initialize(mx.init.Uniform(), ctx=ctx)
        #self.trainer = gluon.Trainer(self.model.collect_params(), 'adadelta', {'rho': 0.9})
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-4})
        return model, trainer

    def train(self, x, v):
        x = nd.array(x)
        v = nd.array(v)

        with autograd.record():
            o = self.model(x)
            L = gluon.loss.L2Loss()
            loss = L(o, v)
            autograd.backward(loss)

        self.trainer.step(batch_size=x.shape[0])

    def predict(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = np.asarray(x)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        return self.model.forward(x).squeeze().asnumpy()

    def load(self, filename):
        self.model.load_params(filename)
        return self

    def save(self, filename):
        try:
            self.model.save_params(filename)
            success = True
        except:
            success = False

        return success
