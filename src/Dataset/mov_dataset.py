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
from matplotlib import pylab as plt
import seaborn as sea
import mpl_toolkits.mplot3d.axes3d as p3
sea.set_style("whitegrid")
from matplotlib import animation
from matplotlib.animation import FuncAnimation

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
	def __init__(self, dataset_dir, list_name):
		"""
		Loads dataset description
		@Arguments:
		dataset_dir: path to the dataset folder with all the data
		list_name: name of the file with the list of entries to include
		"""

		self.targets = []

		with open(os.path.join(dataset_dir, list_name), 'r') as fin:
			for line in fin:
				target_name = line.split()[0]
				self.targets.append(os.path.join(dataset_dir, target_name))
	
		self.dataset_size = len(self.targets)
		
		print "Dataset folder: ", dataset_dir
		print "Dataset list path: ", list_name
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
		torch_x = torch.from_numpy(data["in"].astype('float32'))/100.0
				
		return path, torch_x
		
		
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
		
		

def get_stream(dataset_dir, list_name, batch_size = 10, shuffle = True):
	dataset = WavesDataset(dataset_dir, list_name)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	dataset_dir = os.path.join(DATA_DIR, 'pkls')
	list_name = 'training_set.dat'
	
	dataiter = iter(get_stream(dataset_dir, list_name, 10, False))
	path, x = dataiter.next()

	print 'Input batch size:', x.size()
	x = x[0,:,:,:]
	print torch.max(x), torch.min(x)
	
	fig = plt.figure()
	ax = plt.axes()
	image = ax.imshow(x[0,:,:])
	ttl = ax.set_title('frame %d'%0,animated=True)

	def update_plot(i):
		image.set_data(x[i,:,:])
		ttl.set_text('frame %d'%i)
		return image, ttl

	anim = FuncAnimation(fig, update_plot, frames=x.size(0), blit=True)
	anim.save('tmp.gif', dpi=80, writer='imagemagick')

	
	
	
	

	
