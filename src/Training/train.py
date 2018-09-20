import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import *
from src.Dataset import get_dataset_file
from src.Dataset.waves_dataset import get_stream
from train_loop import train_loop

import torch
import torch.nn as nn


from src.Models import BasicModel

def main(	batch_size = 10,
			cuda_dev = 1,
			experiment_log_dir = None,
			restart = False,
			load_dir = None
		):
	torch.manual_seed(42)
	torch.cuda.set_device(cuda_dev)
	print 'Current device = ', torch.cuda.current_device()

	train_dataset_path = get_dataset_file("training_set")
	train_stream = get_stream(train_dataset_path, batch_size = batch_size, shuffle = False)
	
	validation_dataset_path = get_dataset_file("validation_set")
	validation_stream = get_stream(validation_dataset_path, batch_size = batch_size, shuffle = False)
	
	models_dir_path = os.path.join(MODELS_DIR, experiment_log_dir)
	if not os.path.exists(models_dir_path):
		os.mkdir(models_dir_path)

	full_exp_log_dir = os.path.join(LOG_DIR, experiment_log_dir)
	if not os.path.exists(full_exp_log_dir):
		os.mkdir(full_exp_log_dir)
	
	net = BasicModel()
	#if restart:
	#epoch = 90
	state_dict = torch.load(os.path.join(MODELS_DIR, load_dir, 'net_epoch_start.pth'))
	net.load_state_dict(state_dict)
	

	train_loop( train_dataset_stream = train_stream,
				validation_dataset_stream = validation_stream,
				net = net,
				loss = nn.L1Loss(),
				cuda_dev = cuda_dev,
				learning_rate = 0.0001,
				start_epoch = 0,
				max_epoch = 300,
				batch_size = batch_size,
				model_save_period = 10,
				model_save_dir = models_dir_path,
				logger = full_exp_log_dir)

	
if __name__=='__main__':
	main(	batch_size = 10,
			cuda_dev = 0,
			experiment_log_dir = 'TestCode',
			restart = False,
			load_dir = 'TestCode')
