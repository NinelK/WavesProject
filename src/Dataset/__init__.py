# -*- coding: utf-8 -*-
"""

"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR



def get_dataset_file(dataset_name):
	if dataset_name == 'data_folder':
		return "/home/lupoglaz/Projects/WavesProject/dataset"
	elif dataset_name == "training_set":
		return "/home/lupoglaz/Projects/WavesProject/dataset/training.csv"
	elif dataset_name == "validation_set":
		return "/home/lupoglaz/Projects/WavesProject/dataset/validation.csv"
	else:
		raise IOError("Not found or recognized dataset")