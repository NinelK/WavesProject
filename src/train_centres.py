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
from torch.autograd import Variable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import CENTRES_DIR, MODELS_DIR, LOG_DIR

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein folder')
	parser.add_argument('-experiment', default='CentreTest', help='Experiment name')
	
	parser.add_argument('-image_model', default='Simple', help='Image prediction model')
	parser.add_argument('-dataset_dir', default='', help='Image prediction model')
			
	parser.add_argument('-lr', default=0.001, help='Learning rate', type=float)
	parser.add_argument('-max_epoch', default=300, help='Max epoch', type=int)
	parser.add_argument('-save_interval', default=10, help='Model saving interval in epochs', type=int)

	args = parser.parse_args()

	torch.manual_seed(42)
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
	#loss_model = nn.L1Loss().cuda()

	trainer = CentresTrainer(image_model = image_model,
							loss_model = nn.L1Loss(),
							lr=float(args.lr))
		
	
	#trainer.load_models(epoch, MDL_DIR)
	#image_model.load_state_dict(torch.load(os.path.join(MDL_DIR,'start.th')))
		
	
	data_path = os.path.join(CENTRES_DIR, args.dataset_dir)
	if not os.path.exists(data_path):
		raise(Exception("dataset not found", data_path))

	stream_train = get_stream_centres(data_path, 'training_set.dat')
	stream_valid = get_stream_centres(data_path, 'validation_set.dat')
	
	W = 86
	xx, yy = np.ogrid[:W,:W]
	sel = np.array((xx-(W-1)/2)**2 + (yy-(W-1)/2)**2 < (43)**2).astype("float32")
	mask = torch.from_numpy(sel)
	mask = Variable(mask.cuda())

	for epoch in range(args.max_epoch):
		loss_train = []
		loss_valid = []
		
		trainer.new_log(os.path.join(EXP_DIR,"training_loss%d.dat"%epoch))
		for data in tqdm(stream_train):
			loss_train.append(trainer.optimize(data,mask))
		
		trainer.new_log(os.path.join(EXP_DIR,"validation_loss%d.dat"%epoch))
		for data in tqdm(stream_valid):
			loss_valid.append(trainer.predict(data,mask))
		
		print('Loss train = %f\n Loss valid = %f\n'%(np.mean(loss_train), np.mean(loss_valid)))

		if (epoch+1)%args.save_interval==0:
			trainer.save_models(epoch, MDL_DIR)
		
