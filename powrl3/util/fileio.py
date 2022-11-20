#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Util functions for I/O"""

__author__ = 'leferrad'

import json
import logging
import logging.handlers
import os
import pickle
import tarfile


def get_logger(name='powrl3', level='debug'):
    """
    Function to obtain a normal logger

    :param name: string
    :param level: string, which can be 'info' or 'debug'
    :return: logging.Logger
    """

    levels = {'info': logging.INFO,
              'debug': logging.DEBUG}

    # If the level is not supported, then force it to be info
    if level not in levels:
        level = 'info'
    level = levels[level]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


def serialize_python_object(obj, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        successful = True
    except:
        successful = False
    return successful


def deserialize_python_object(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    except:
        obj = None
    return obj


def save_dict_as_json(dictobj, filename, pretty_print=True):
    try:
        with open(filename, 'w') as f:
            if pretty_print is True:
                json.dump(dictobj, f, sort_keys=True, indent=4)
            else:
                json.dump(dictobj, f)
        successful = True
    except:
        successful = False
    return successful


def load_json_as_dict(filename):
    try:
        with open(filename, 'r') as f:
            dictobj = json.load(f)
    except:
        dictobj = None
    return dictobj


def compress_tar_files(files, filename):
    if isinstance(files, list) is False:
        files = [files]
    try:
        with tarfile.open(filename, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=os.path.basename(f))
        successful = True
    except:
        successful = False
    return successful


def decompress_tar_files(filename):
    try:
        with tarfile.open(filename, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=os.path.dirname(filename))
        successful = True
    except:
        successful = False
    return successful
