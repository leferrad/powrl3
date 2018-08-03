#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit-testing for environment functions"""

from __future__ import print_function

from powrl3.agent.feature import LBPFeatureTransformer
from powrl3.environment.cube import CubeEnvironment

import matplotlib.pyplot as plt
import numpy as np


__author__ = 'leferrad'


def main_env():
    """
    Functional testing.
    # TODO: this should be a unit test
    """
    # Taking all of the supported actions

    seed = np.random.randint(0, 100)
    print("Using seed=%i" % seed)
    ce = CubeEnvironment()
    print("---o- ---------------------------- -o---")
    print("---o- Taking all supported actions -o---")
    print("---o- ---------------------------- -o---")
    state = ce.cube.get_state()
    print("State: %s" % str(state))
    lbp_code = LBPFeatureTransformer.transform(state)
    print("LBP code: %s" % str(lbp_code))
    lbp_hist = LBPFeatureTransformer.hist_lbp_code(lbp_code)
    print("LBP hist: %s" % str(lbp_hist))
    print("It's solved!" if ce.is_solved() else "Not solved!")
    for a in CubeEnvironment.actions_available:
        print("Taking the following action: %s" % a)
        ce.take_action(a)
        state = ce.cube.get_state()
        print("State: %s" % str(state))
        lbp_code = LBPFeatureTransformer.transform(state)
        print("LBP code: %s" % str(lbp_code))
        lbp_hist = LBPFeatureTransformer.hist_lbp_code(lbp_code)
        print("LBP hist: %s" % str(lbp_hist))
        print("It's solved!" if ce.is_solved() else "Not solved!")
        # ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)

    print("Algorithm followed: %s" % ce.actions_taken)

    # Now, let's take some random actions

    n_random_actions = 10
    print("---o- ------------------------ -o---")
    print("---o- Taking %i random actions -o---" % n_random_actions)
    print("---o- ------------------------ -o---")
    ce = CubeEnvironment()

    for m in range(n_random_actions):
        print("State: %s" % str(ce.cube.get_state()))
        print("It's solved!" if ce.is_solved() else "Not solved!")
        # ce.render(flat=False)#.savefig("test%02d.png" % m, dpi=865 / c.N)
        a = np.random.choice(CubeEnvironment.actions_available)
        print("Taking the following action: %s" % a)
        ce.take_action(a)

    print("Algorithm followed: %s" % ce.actions_taken)

    plt.show()
