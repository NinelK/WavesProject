import os
import sys
import torch
import argparse
from Training import CentresTrainer
from Models import CentresModel
from Dataset import get_stream_centres
from tqdm import tqdm
from torch import nn
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import CENTRES_DIR, MODELS_DIR, LOG_DIR

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein folder')
	parser.add_argument('-experiment', default='CentreTest', help='Experiment name')
	
	parser.add_argument('-image_model', default='Simple', help='Image prediction model')
	parser.add_argument('-dataset_dir', default='', help='Image prediction model')
			
	parser.add_argument('-load_epoch', default=None, help='Max epoch', type=int)
	

	args = parser.parse_args()

	torch.cuda.set_device(1)
	torch.backends.cudnn.enabled = False
	
	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	MDL_DIR = os.path.join(MODELS_DIR, args.experiment)
	try:
		os.mkdir(EXP_DIR)
	except:
		pass
	try:
		os.mkdir(MDL_DIR)
	except:
		pass

	
	image_model = CentresModel().cuda()

	trainer = CentresTrainer(	image_model = image_model,
							loss_model = nn.L1Loss(),
							lr=0.0)
		
	epoch = 0
	if args.load_epoch is None:
		for filename in os.listdir(os.path.join(MDL_DIR)):
			if filename.find('epoch')!=-1:
				epoch_num = filename[filename.find('epoch') + len('epoch'):filename.rfind('.')]
				if int(epoch_num)>epoch:
					epoch = int(epoch_num)
		trainer.load_models(epoch, MDL_DIR)
	else:
		trainer.load_models(args.load_epoch, MDL_DIR)
		epoch = args.load_epoch
	print('Loaded from epoch = ', epoch)
	
	data_path = os.path.join(CENTRES_DIR, args.dataset_dir)
	if not os.path.exists(data_path):
		raise(Exception("dataset not found", data_path))
	
	stream_valid = get_stream_centres(data_path, 'video_set.dat')
	
	trainer.new_log(os.path.join(EXP_DIR,"test_loss.dat"), log_dir=os.path.join(EXP_DIR,'video'))
	for data in tqdm(stream_valid):
		trainer.predict(data)
		
