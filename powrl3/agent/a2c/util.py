#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from powrl3.util.fileio import *

import numpy as np

from collections import deque
import random
import pickle
import os


class EligibilityTraces(object):
    def __init__(self, actions, n_dim):
        self.actions = actions
        n = len(actions)
        self.n_dim = n_dim
        self.traces = np.zeros((n, n_dim))

    def reset(self):
        self.traces = np.zeros_like(self.traces)

    def get(self, a):
        return self.traces[self.actions[a]]

    def update(self, X, a, gamma=0.99, lambda_=0.3):
        self.traces *= gamma*lambda_
        i = self.actions[a]
        self.traces[i] += X

    def save(self, filename):
        """
        Save the model in a TAR file
        :param filename: string, path to file where store the model
        >>> # Load
        >>> model_path = '/tmp/model/el_tr.tgz'
        >>> test_model = EligibilityTraces.load(filename=model_path)
        >>> # Save
        >>> test_model.save(filename=model_path)
        """
        # 1) Save the JSON with actions (to know which vector corresponds with each action)

        base_path = os.path.dirname(filename)
        config_path = os.path.join(base_path, 'config.json')
        traces_path = os.path.join(base_path, 'traces.pkl')

        actions = [{"action": a, "index": i} for (a, i) in self.actions.items()]
        json_res = {"n_dimensions": self.n_dim, "actions": actions}
        success_save_config = save_dict_as_json(json_res, filename=config_path, pretty_print=True)

        # 2) Save the serialized array of traces
        success_save_traces = serialize_python_object(self.traces, traces_path)

        if all([success_save_config, success_save_traces]):
            files = [config_path, traces_path]
            success = compress_tar_files(files=files, filename=filename)
        else:
            success = False

        try:
            # remove temporary files which are already compressed
            os.remove(config_path)
            os.remove(traces_path)
        except:
            pass

        return success

    def load(self, filename):
        success = False
        success_decompress = decompress_tar_files(filename)
        if success_decompress is True:
            base_path = os.path.dirname(filename)
            config_path = os.path.join(base_path, 'config.json')
            traces_path = os.path.join(base_path, 'traces.pkl')

            # 1) Load the JSON config file
            json_res = load_json_as_dict(config_path)

            # TODO: validate json loaded

            self.n_dim = json_res["n_dimensions"]
            self.actions = dict([(r["action"], r["index"])
                                for r in json_res["actions"]])

            # 2) Load the array of traces
            try:
                with open(traces_path, 'rb') as f:
                    self.traces = pickle.load(f)
                success = True
            except:
                self.traces = np.zeros((len(self.actions), self.n_dim))
                success = False

        try:
            # remove temporary files which are already decompressed
            os.remove(config_path)
            os.remove(traces_path)
        except:
            pass

        return self


class ReplacingEligibilityTraces(EligibilityTraces):
    def __init__(self, actions, n_dim):
        EligibilityTraces.__init__(self, actions=actions, n_dim=n_dim)

    def update(self, X, a, gamma=0.99, lambda_=0.3):
        self.traces *= gamma*lambda_
        i = self.actions[a]
        self.traces[i] = X


# - https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
# - https://pymotw.com/2/collections/deque.html

class ExperienceReplay2(object):
    def __init__(self, max_items=5000):
        self.buffer = deque(maxlen=max_items)
        self.max_items = max_items

    def add_state(self, state, value):
        # Overwrite existing value if the state is already stored
        self.buffer.append((state, value))

    def get_sample(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        return random.sample(self.buffer, n_items)

    def save(self, filename):
        """
        Save the model in a serialized file
        :param filename: string, path to file where store the model
        >>> # Load
        >>> model_path = '/tmp/model/test_er.json'
        >>> test_model = ExperienceReplay().load(filename=model_path)
        >>> # Save
        >>> test_model.save(filename=model_path)
        """
        buf = [{"state": k, "value": v} for (k, v) in self.buffer]
        json_res = {"max_items": self.max_items, "buffer": buf}
        success = save_dict_as_json(json_res, filename=filename, pretty_print=True)

        return success

    def load(self, filename):
        json_res = load_json_as_dict(filename)

        # TODO: validate json loaded

        self.max_items = json_res["max_items"]
        self.buffer = deque([(b["state"], b["value"])
                            for b in json_res["buffer"]])

        return self

    def __len__(self):
        return len(self.buffer)


class ExperienceReplay(object):
    def __init__(self, max_items=5000):
        self.buffer = {}
        self.max_items = max_items

    def add_state(self, state, value):
        # Overwrite existing value if the state is already stored
        self.buffer[state] = value

        if len(self.buffer) > self.max_items:
            # Forget the min value
            pop_item = min(self.buffer.items(), key=lambda i: i[1])
            # Forget a random value
            #pop_item = random.choice(self.buffer.items())
            self.buffer.pop(pop_item[0])

    def get_sample(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        return random.sample(self.buffer.items(), n_items)

    def save(self, filename):
        """
        Save the model in a serialized file
        :param filename: string, path to file where store the model
        >>> # Load
        >>> model_path = '/tmp/model/test_er.json'
        >>> test_model = ExperienceReplay().load(filename=model_path)
        >>> # Save
        >>> test_model.save(filename=model_path)
        """
        buf = [{"state": k, "value": v} for (k, v) in self.buffer.items()]
        json_res = {"max_items": self.max_items, "buffer": buf}
        success = save_dict_as_json(json_res, filename=filename, pretty_print=True)

        return success

    def load(self, filename):
        json_res = load_json_as_dict(filename)

        # TODO: validate json loaded

        self.max_items = json_res["max_items"]
        self.buffer = dict([(b["state"], b["value"])
                            for b in json_res["buffer"]])

        return self

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayAC(object):
    def __init__(self, max_items=5000):
        self.buffer = {}
        self.max_items = max_items

    def add(self, s, a, r, s_prime, t):
        # Overwrite existing value if the state is already stored
        self.buffer[tuple(s)] = (a, r, s_prime, t)

        if len(self.buffer) > self.max_items:
            # Forget the item with the least reward
            #pop_item = min(self.buffer.items(), key=lambda i: i[1][1])
            pop_item = min(self.buffer.items(), key=lambda i: abs(i[1][1]))
            # Forget a random value
            #pop_item = random.choice(self.buffer.items())
            self.buffer.pop(pop_item[0])

    def get_sample(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        sample = [(s, a, r, s_prime, t) for (s, (a, r, s_prime, t)) in random.sample(self.buffer.items(), n_items)]
        return sample

    def get_sample2(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        population = self.buffer.items()
        sample_index = np.random.multinomial(n=n_items, pvals=[])
        sample = [(s, a, r, s_prime, t) for (s, (a, r, s_prime, t)) in random.sample(population, n_items)]
        return sample

    def save(self, filename):
        """
        Save the model in a serialized file
        :param filename: string, path to file where store the model
        >>> # Load
        >>> model_path = '/tmp/model/test_er.json'
        >>> test_model = ExperienceReplay().load(filename=model_path)
        >>> # Save
        >>> test_model.save(filename=model_path)
        """
        buf = [{"state": k, "value": v} for (k, v) in self.buffer.items()]
        json_res = {"max_items": self.max_items, "buffer": buf}
        success = save_dict_as_json(json_res, filename=filename, pretty_print=True)

        return success

    def load(self, filename):
        json_res = load_json_as_dict(filename)

        # TODO: validate json loaded

        self.max_items = json_res["max_items"]
        self.buffer = dict([(b["state"], b["value"])
                            for b in json_res["buffer"]])

        return self

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayACA(object):
    def __init__(self, max_items=5000):
        self.buffer = {}
        self.max_items = max_items

    def add(self, s, a, r, s_prime, t):
        # Overwrite existing value if the state is already stored
        self.buffer[(tuple(s), a)] = (r, s_prime, t)

        if len(self.buffer) > self.max_items:
            # Forget the item with the least reward
            #pop_item = min(self.buffer.items(), key=lambda i: i[1][0])
            pop_item = min(self.buffer.items(), key=lambda i: abs(i[1][0]))
            # Forget a random value
            #pop_item = random.choice(self.buffer.items())
            self.buffer.pop(pop_item[0])

    def get_sample(self, max_items=50):
        n_items = min(len(self.buffer), max_items)
        sample = [(s, a, r, s_prime, t) for ((s, a), (r, s_prime, t)) in random.sample(self.buffer.items(), n_items)]
        return sample

    def save(self, filename):
        """
        Save the model in a serialized file
        :param filename: string, path to file where store the model
        >>> # Load
        >>> model_path = '/tmp/model/test_er.json'
        >>> test_model = ExperienceReplay().load(filename=model_path)
        >>> # Save
        >>> test_model.save(filename=model_path)
        """
        buf = [{"state": k, "value": v} for (k, v) in self.buffer.items()]
        json_res = {"max_items": self.max_items, "buffer": buf}
        success = save_dict_as_json(json_res, filename=filename, pretty_print=True)

        return success

    def load(self, filename):
        json_res = load_json_as_dict(filename)

        # TODO: validate json loaded

        self.max_items = json_res["max_items"]
        self.buffer = dict([(b["state"], b["value"])
                            for b in json_res["buffer"]])

        return self

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayEligibility(ExperienceReplay):
    def __init__(self, max_items=5000):
        ExperienceReplay.__init__(self, max_items=max_items)

    def add_state_e(self, state, value, e):
        # Overwrite existing value if the state is already stored
        self.buffer[state] = (value, e)

        if len(self.buffer) > self.max_items:
            # Forget the min value
            pop_item = min(self.buffer.items(), key=lambda i: i[1][0])
            # Forget a random value
            #pop_item = random.choice(self.buffer.items())
            self.buffer.pop(pop_item[0])

    def save(self, filename):
        buf = [{"key": k, "value": v, "eligibility": list(e)}
               for (k, (v, e)) in self.buffer.items()]
        json_res = {"max_items": self.max_items, "buffer": buf}
        success = save_dict_as_json(json_res, filename=filename, pretty_print=True)

        return success

    def load(self, filename):
        json_res = load_json_as_dict(filename)

        # TODO: validate json loaded

        self.max_items = json_res["max_items"]
        self.buffer = dict([(tuple(b["key"]), (b["value"], np.asarray(b["eligibility"])))
                            for b in json_res["buffer"]])

        return self
