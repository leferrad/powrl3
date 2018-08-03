#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""""""

__author__ = 'leferrad'

from powrl3.environment.cube import CubeEnvironment
from powrl3.agent.feature import LBPFeatureTransformer

import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":
    """
    Let's take some random actions for a couple of iterations to figure out the space of possible states
    """
    n_random_actions = 100
    iters = 20
    seen_states = {}

    for i in range(iters):
        seed = np.random.randint(0, 100)
        print("---o- ------------------------ -o---")
        print("---o-     Iteration N° %i      -o---" % i)
        print("---o- ------------------------ -o---")
        print("---o- Taking %i random actions -o---" % n_random_actions)
        print("---o- ------------------------ -o---")
        print("Using seed=%i" % seed)
        ce = CubeEnvironment(seed=seed)
        for m in range(n_random_actions):
            action = ce.sample_action()
            print("Taking the following action: %s" % action)
            ce.take_action(action)
            state = ce.get_state()
            print("State: %s" % str(state))
            lbp_code = LBPFeatureTransformer.transform(state)
            print("LBP code: %s" % str(lbp_code))
            # Adding state to the set
            lbp_code = tuple(lbp_code)
            if lbp_code in seen_states:
                seen_states[lbp_code] += 1
            else:
                seen_states[lbp_code] = 1

            # Just to discover some interesting patterns...
            if ce.is_solved():
                print("WOW! The cube is solved!")
                time.sleep(5)
            print(ce.render())

        print("Algorithm followed: %s" % ce.actions_taken)

    plt.show()
    print("There were %i different seen states after taking a total of %i actions!" % \
          (len(seen_states), iters*n_random_actions))
    print("States: %s" % str(sorted(seen_states.items(), key=lambda x: -x[1])))