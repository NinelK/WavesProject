# -*- coding: utf-8 -*-
"""
Splitting the dataset into validation and training subsets
"""

import itertools
import logging
import os
import sys
import torch
from torch.utils.data import Dataset
import atexit
import numpy as np
import cPickle as pkl

from os import listdir
from os.path import isfile
import random
random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Dataset import get_dataset_file

logger = logging.getLogger(__name__)

if __name__=='__main__':
    data_folder = get_dataset_file("data_folder")
    training_list_path = get_dataset_file("training_set")
    validation_list_path = get_dataset_file("validation_set")
    data_list = listdir(data_folder)
    
    fraction = 0.8
    last_training_idx = int(np.floor(len(data_list)*fraction))
    training_list = data_list[:last_training_idx]
    validation_list = data_list[last_training_idx:]
    
    with open(training_list_path, 'w') as fout:
        for entry in training_list:
            fout.write(entry+'\n')

    with open(validation_list_path, 'w') as fout:
        for entry in validation_list:
            fout.write(entry+'\n')
