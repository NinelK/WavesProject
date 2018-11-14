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
from PIL import Image
# import pickle as pkl
import _pickle as pkl
import matplotlib.pylab as plt

from os import listdir
from os.path import isfile
import random
random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR

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
		
		with open(os.path.join(self.dataset_dir, self.list_path), 'r') as fin:
			for line in fin:
				target_name = line.split()[0]
				self.targets.append(os.path.join(self.dataset_dir, target_name) )
		
		
		self.dataset_size = len(self.targets)
		
		print("Dataset folder: ", self.dataset_dir)
		print("Dataset list path: ", self.list_path)
		print("Dataset size: ", self.dataset_size)
		print("Dataset output type: 3d density maps")

		atexit.register(self.cleanup)
		
	def __getitem__(self, index):
		"""
		Returns data path, 3d array of data and 2d array answer
		"""
		path = self.targets[index]
		if path.split('.')[-1] == 'pkl':
			with open(path, 'rb') as fin:
				data = pkl.load(fin, encoding='latin1')
			# data = pkl.load(path)
		
			data_size = data["in"].shape
			torch_x = torch.from_numpy(data["in"].reshape((data_size[0], data_size[1], data_size[2])).astype('float32'))
			torch_x = torch_x/torch.max(torch_x)
		
			#data_size = data["out"].shape
			#torch_y = torch.from_numpy(data["out"].reshape((data_size[0], data_size[1], data_size[2])).astype('float32'))
			#torch_y = torch_y/torch.max(torch_y)
		
			return path.split('/')[-1].split('.')[0], torch_x[0,:,:], torch_x[0,:,:]
		elif path.split('.')[-1] == 'png':
			with open(path, 'rb') as fin:
				data = np.asarray(Image.open(fin))
		
			data_size = data.shape
			#print(data_size)
			torch_x = torch.from_numpy(data.reshape((data_size[0], data_size[1])).astype('float32'))
			torch_x = torch_x/torch.max(torch_x)
		
			torch_y = torch.from_numpy(data.reshape((data_size[0], data_size[1])).astype('float32'))
			torch_y = torch_x/torch.max(torch_x)
			
			return path.split('/')[-1].split('.')[0], torch_x, torch_y
		else:
			print('Wrong file format (pkl or png is needed).')
			return 0

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
		
		

def get_stream_vae(dataset_dir, list_name, batch_size = 10, shuffle = True):
	dataset = WavesDataset(dataset_dir, list_name)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	
	dataset_dir = os.path.join(DATA_DIR, 'data5', 'pkls')
	list_name = 'training_set.dat'
	dataiter = iter(get_stream_vae(dataset_dir, list_name, 10, False))
	path, x = dataiter.next()

	print('Input batch size:', x.size())
	
	plt.imshow(x[0].numpy())
	plt.colorbar()
	plt.show()

	stream = get_stream_vae(dataset_dir, list_name, 10, False)
	for data in stream:
		pass
		
	
	
	

	
