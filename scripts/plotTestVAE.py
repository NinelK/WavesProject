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

def plot_gif(grnd, pred, filename):
	fig = plt.figure()
	ax = plt.axes()
	data = torch.cat([grnd[0,:,:],pred[0,:,:]], dim=1).data.numpy()
	
	image = ax.imshow(data)
	ttl = ax.set_title('frame %d'%0,animated=True)

	def update_plot(i):
		data = torch.cat([grnd[i,:,:],pred[i,:,:]], dim=1).data.numpy()
		print torch.mean(pred[i,:,:]), torch.mean(grnd[i,:,:])
		image.set_data(data)
		ttl.set_text('frame %d'%i)
		return image, ttl

	anim = FuncAnimation(fig, update_plot, frames=grnd.size(0), blit=True)
	anim.save(filename, dpi=80, writer='imagemagick')

def plot_gifs(experiment_name='Test', test_name='test'):
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
	
	for sample_name in grnd.keys():
		plot_gif(grnd[sample_name], pred[sample_name], sample_name+'.gif')


if __name__=='__main__':
	experiment_name='VAETest'
	plot_gifs(experiment_name=experiment_name, test_name='train')

