import os
import sys
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
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
import scipy.misc

def plot_fig(grnd, pred, filename):
	fig = plt.figure()
	ax = plt.axes()

	LossModel = torch.nn.L1Loss(size_average=True)
	#LossModel = torch.nn.BCELoss(size_average=True)
	#size = grnd.size()
	loss = LossModel(pred,grnd).data.numpy()

	#f = torch.cat([grnd,pred], dim=0).data.numpy()
	#image = ax.imshow(f)
	#plt.text(0,0,"Loss: %.4f" % loss)
	#plt.savefig("%d_%s" % (np.round(loss*1000)-500,filename))
	#plt.show()

	scipy.misc.imsave(filename,pred.data.numpy())

	return loss

def plot_figs(experiment_name='Test', test_name='test'):
	log_dir = os.path.join(LOG_DIR, experiment_name, test_name)
	pred = {}
	grnd = {}
	for filename in os.listdir(log_dir):
		log_type = filename.split('_')[1].split('.')[0]
		sample_name = filename.split('_')[0]

		if log_type == 'grnd':
			grnd[sample_name] = torch.load(os.path.join(log_dir,filename))
		elif log_type == 'pred':
			pred[sample_name] = torch.load(os.path.join(log_dir,filename))
		else:
			continue

	max = 0.0
	min = 1.0 #normalised Loss < 1
	for sample_name in tqdm(grnd.keys()):
		loss = plot_fig(grnd[sample_name], pred[sample_name], sample_name+'.png')
		#print(loss)
		if loss < min:
			path_min = sample_name
			min = loss
		if loss > max:
			path_max = sample_name
			max = loss
	
	print("Max loss: %.4f for %s" % (max,path_max))
	print("Min loss: %.4f for %s" % (min,path_min))


if __name__=='__main__':
	experiment_name='CentreTest'
	plot_figs(experiment_name=experiment_name, test_name='video')

