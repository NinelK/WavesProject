import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib import pylab as plt
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import animation
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, DATA_DIR
import torch

def read_log(filename):
	loss = []
	with open(filename) as fin:
		for line in fin:
			sline = line.split()
			if sline[0] == 'Loss':
				loss.append(float(sline[1]))
			else:
				raise(Exception('Unknown file'))
	return np.mean(loss)

def dict2list(dict):
	return [dict[i] for i in xrange(max(dict.keys()))]

def plot_training_loss(experiment_name='Test', filename='training_loss.dat'):
	log_dir = os.path.join(LOG_DIR, experiment_name)
	loss_train = {}
	loss_valid = {}

	for filename in os.listdir(log_dir):
		if filename.find('loss')==-1:
			continue
		
		log_type = filename.split('_')[0]
		
		if log_type == 'training':
			epoch = int(filename.split('loss')[1].split('.')[0])
			loss_train[epoch] = read_log(os.path.join(log_dir,filename))
		elif log_type == 'validation':
			epoch = int(filename.split('loss')[1].split('.')[0])
			loss_valid[epoch] = read_log(os.path.join(log_dir,filename))
		else:
			continue
	
	loss_train = dict2list(loss_train)
	loss_valid = dict2list(loss_valid)
	
	plt.figure(figsize=(12,8))
	plt.plot(np.array(loss_train), label='train')
	plt.plot(np.array(loss_valid), label='valid')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.show()

if __name__=='__main__':
	experiment_name='VAETest'
	plot_training_loss(experiment_name=experiment_name)

