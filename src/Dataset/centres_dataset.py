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
import scipy.ndimage

from os import listdir
from os.path import isfile
import random
random.seed(42)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import CENTRES_DIR

logger = logging.getLogger(__name__)

class CentresDataset(Dataset):
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
			gradT,gradX,gradY = np.mgrid[0:data_size[0],0:data_size[1],0:data_size[2]]/np.max(data_size) #normalized
			W = 86
			xx, yy = np.ogrid[:W,:W]
			selF = np.uint8( (xx-(W-1)/2.0)**2 + (yy-(W-1)/2.0)**2 < (W/2.0)**2 )
			data_mov = data["in"]+np.random.normal(0, 1, size=data["in"].shape)*0.05*np.max(data["in"])
			data_mov = (data_mov-np.min(data_mov))/(np.max(data_mov)-np.min(data_mov))*selF
			IN = np.array([data_mov,gradT,gradX,gradY])

			if(np.max(data_mov) > 1.0):
				print("Warning! Data out of [0,1] range!")

			torch_x = torch.from_numpy(IN.reshape((4,data_size[0], data_size[1], data_size[2])).astype('float32'))
			#torch_x = torch_x/torch.max(torch_x[0])
		
			data_size = data["out"].shape
			data["out"] = (np.max(data["out"])==data["out"])*1.0
			#print(np.max(data["out"]))
			oreol=scipy.ndimage.gaussian_filter(data["out"], sigma=1)
			oreol = 1.0 * oreol/np.max(oreol)
			data["out"]= data["out"]*0.0+oreol
			torch_y = torch.from_numpy(data["out"].reshape((data_size[0], data_size[1])).astype('float32'))
			torch_y = torch_y/torch.max(torch_y)
		
			return path.split('/')[-1].split('.')[0], torch_x, torch_y

		elif path.split('.')[-1] == 'png':
			with open(path, 'rb') as fin:
				data = np.asarray(Image.open(fin))
		
			data_size = data.shape
			gradT,gradX,gradY = np.mgrid[0:data_size[0],0:data_size[1],0:data_size[2]]/np.max(data_size) #normalized
			data = data/np.max(data)
			IN = np.array([data,gradT,gradX,gradY])

			if(np.max(data) > 1.0):
				print("Warning! Data out of [0,1] range!")

			torch_x = torch.from_numpy(IN.reshape((4, data_size[0], data_size[1], data_size[2])).astype('float32'))
			#torch_x = torch_x/torch.max(torch_x[0])

			torch_y = torch.from_numpy(data.reshape((data_size[0], data_size[1], data_size[2])).astype('float32'))
			torch_y = torch_y/torch.max(torch_y)
			
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
		
		

def get_stream_centres(dataset_dir, list_name, batch_size = 100, shuffle = True):
	dataset = CentresDataset(dataset_dir, list_name)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	
	dataset_dir = os.path.join(CENTRES_DIR)
	list_name = 'training_set.dat'
	dataiter = iter(get_stream_centres(dataset_dir, list_name, 10, False))
	path, x = dataiter.next()

	print('Input batch size:', x.size())
	
	plt.imshow(x[0].numpy())
	plt.colorbar()
	plt.show()

	stream = get_stream_centres(dataset_dir, list_name, 10, False)
	for data in stream:
		pass
		
	
	
	

	
