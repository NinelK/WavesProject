# -*- coding: utf-8 -*-
"""
Dataset wrapper around coordinates hdf5 files. Returns pytorch::DataLoader stream.
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
import matplotlib.pylab as plt

from os import listdir
from os.path import isfile
import random
random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Dataset import get_dataset_file

logger = logging.getLogger(__name__)

class WavesDataset(Dataset):
	"""
	The class that samples the data
	"""
	def __init__(self, dataset_dir, list_path):
		"""
		Loads dataset description
		@Arguments:
		dataset_dir: path to the dataset folder with all the data
		list_path: path to the csv file with the list of entries to include
		"""

		self.dataset_dir = dataset_dir
		self.list_path = list_path
		self.targets = []

		with open(self.list_path, 'r') as fin:
			for line in fin:
				target_name = line.split()[0]
				self.targets.append(os.path.join(self.dataset_dir, target_name))
	
		self.dataset_size = len(self.targets)
		
		print "Dataset folder: ", self.dataset_dir
		print "Dataset list path: ", self.list_path
		print "Dataset size: ", self.dataset_size
		print "Dataset output type: 3d density maps"

		atexit.register(self.cleanup)
		
	def __getitem__(self, index):
		"""
		Returns data path, 3d array of data and 2d array answer
		"""
		path = self.targets[index]
		with open(path, 'r') as fin:
			data = pkl.load(fin)
		data_size = data["in"].shape
		torch_x = torch.from_numpy(data["in"].reshape((1,data_size[0], data_size[1], data_size[2])).astype('float32'))
		torch_y = torch.from_numpy(data["out"].astype('float32'))
		
		return path, torch_x, torch_y
		
		
	def __len__(self):
		"""
		Returns length of the dataset
		"""
		return self.dataset_size

	def cleanup(self):
		"""
		Closes hdf5 file
		"""
		pass
		
		

def get_stream(data_list_path, batch_size = 10, shuffle = True):
	data_folder = get_dataset_file('data_folder')
	dataset = WavesDataset(data_folder, data_list_path)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	
	data_list_path = get_dataset_file('training_set')
	dataiter = iter(get_stream(data_list_path, 10, False))
	path, x, y = dataiter.next()

	print 'Input batch size:', x.size()
	print 'Output batch size', y.size()

	plt.imshow(y[0].numpy())
	plt.show()
	
	
	
	

	