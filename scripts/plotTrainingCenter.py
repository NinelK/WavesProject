import os
import sys
import numpy as np
# from scipy.stats import pearsonr, spearmanr
from matplotlib import pylab as plt
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import animation
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, DATA_DIR
from src.Dataset import get_image
import torch

def read_loss_file(filename):
	loss = []
	with open(filename) as fin:
		for line in fin:
			loss.append(float(line.split()[1]))
	return np.average(loss)

def read_file_list(experiment_name='Test', tr_name='training_loss', vl_name='validation_loss'):
	dir_name = os.path.join(LOG_DIR, experiment_name)
	max_epoch = 0 
	filenames = {}
	for filename in os.listdir(dir_name):
		if filename.find(tr_name)==-1:
			continue
		epoch = int(filename.split(tr_name)[1].split('.')[0])
		tr_filename = os.path.join(dir_name, tr_name+'%d.dat'%epoch)
		vl_filename = os.path.join(dir_name, vl_name+'%d.dat'%epoch)
		if not os.path.exists(vl_filename):
			continue
		
		filenames[epoch] = (tr_filename, vl_filename)
		if epoch>max_epoch:
			max_epoch = epoch
	return max_epoch, filenames

def plot_training_loss(experiment_name='Test', tr_name='training_loss', vl_name='validation_loss'):
	max_epoch, filenames = read_file_list(experiment_name, tr_name, vl_name)
	val_loss = []
	tr_loss = []
	for epoch in xrange(max_epoch):
		tr_loss.append(read_loss_file(filenames[epoch][0]))
		val_loss.append(read_loss_file(filenames[epoch][1]))
		
	plt.figure(figsize=(12,8))
	plt.plot(np.array(tr_loss), label='training')
	plt.plot(np.array(val_loss), label='validation')
	plt.legend()
	plt.show()

def plot_validation_imgs(experiment_name='Test', epoch = 49):
	pass
	
	

if __name__=='__main__':
	experiment_name='CenterTest'
	plot_training_loss(experiment_name=experiment_name)
	plot_validation_imgs(experiment_name=experiment_name, epoch = 49)