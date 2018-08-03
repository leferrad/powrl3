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
- https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
"""

import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon

from mxnet import nd, autograd
from mxnet.gluon import nn

# TODO: use minpy?
import numpy as np


class PolicyGradientLoss(gluon.loss.Loss):
    def __init__(self,  weight=1., batch_axis=0, **kwargs):
        super(PolicyGradientLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, FF, output, returns, action_prob, sample_weight=None):
        action_prob = FF.sum(output * action_prob, axis=1)
        log_action_prob = FF.log(action_prob)
        loss = -FF.sum(log_action_prob * returns)

        return loss


class Net(gluon.Block):
    def __init__(self, n_dims, n_actions, hidden_dims=(32, 32), **kwargs):
        super(Net, self).__init__(**kwargs)

        self.n_dims = n_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims

        with self.name_scope():

            self.embedding = nn.Embedding(256, output_dim=16)  # suggestion: not greater than 16
            self.bn = nn.BatchNorm()  # TODO: is this necessary?
            self.conv1 = nn.Conv1D(channels=32, kernel_size=3, activation='relu',
                                   padding=0, strides=1)
            self.conv2 = nn.Conv1D(channels=32, kernel_size=3, activation='relu',
                                   padding=0, strides=1)

            self.pool = nn.GlobalMaxPool1D()

            self.h1 = nn.Dense(32, activation='relu')
            self.h2 = nn.Dense(32, activation='relu')

            #for h_dim in self.hidden_dims:
            #    x = Dense(h_dim, activation='relu')(x)
                # x = Dropout(0.2)(x)

            self.output = nn.Dense(self.n_actions, use_bias=False)

    def forward(self, x):
        x = self.embedding(nd.array(x))
        x = self.bn(x)

        x = self.pool(self.conv2(self.conv1(x)))

        x = self.h2(self.h1(x))

        return F.softmax(self.output(x))


class PolicyGradientModel(object):
    # TODO: use eligibility traces as it is explained here:
    # http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf

    def __init__(self, actions, n_dims, lr=1e-4, ctx=mx.cpu(0), use_boltzmann_exploration=True):
        self.actions = actions
        self.n_actions = len(self.actions)
        self.n_dims = n_dims
        self.lr = lr
        self.ctx = ctx

        self.use_boltzmann_exploration = use_boltzmann_exploration
        if use_boltzmann_exploration:
            self.t_boltzmann = 100.0
            self.n_steps = 1

        self.actions_predicted = []

        self.model, self.trainer = self._build_network(n_dims=self.n_dims, n_actions=self.n_actions, ctx=self.ctx)

        #self.model.summary()

    @staticmethod
    def _build_network(n_dims, n_actions, hidden_dims=(32, 32), ctx=mx.cpu(0)):
        model = Net(n_dims=n_dims, n_actions=n_actions, hidden_dims=hidden_dims)
        model.initialize(mx.init.Uniform(), ctx=ctx)
        #self.trainer = gluon.Trainer(self.model.collect_params(), 'adadelta', {'rho': 0.9})
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-4})
        #self.trainer = gluon.Trainer(self.model.collect_params(), 'ftml', {'beta1': 0.6, 'beta2': 0.99})

        return model, trainer

    def train(self, x, a, r):
        x = nd.array(x)
        try:
            a = nd.array(a)
        except TypeError:
            pass
        r = nd.array(r)

        with autograd.record():
            o = self.model(x)
            L = PolicyGradientLoss()
            loss = L(o, r, a)
            autograd.backward(loss)

        self.trainer.step(batch_size=x.shape[0])

    def predict(self, x):
        x = np.expand_dims(x, axis=0)
        return self.model.forward(x).squeeze().asnumpy()

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
            #probability_actions = self.get_boltzmann_values(x)
            probability_actions = self.predict(x)
            a = np.random.choice([t[0] for t in actions], p=probability_actions)
        else:
            # Exploit
            probability_actions = self.predict(x)
            probability_actions = probability_actions.squeeze().asnumpy()
            a_i, p = max(enumerate(probability_actions), key=lambda t: t[1])
            a, i = filter(lambda t: t[1] == a_i, actions)[0]

        # print("Probability actions: %s" % str(probability_actions))

        return a

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
