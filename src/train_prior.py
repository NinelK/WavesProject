import os
import sys
import torch
import argparse
from Training import PriorTrainer
from Models import ImageModelSimple, ResNet, dcgan
from Dataset import get_image
from tqdm import tqdm
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import DATA_DIR, MODELS_DIR, LOG_DIR

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein folder')
	parser.add_argument('-experiment', default='ResNetSpiral5', help='Experiment name')
	
	parser.add_argument('-image_model', default='ResNet', help='Image prediction model')
	parser.add_argument('-image_name', default='SpiralsInPetri/Spiral5_circ.tif', help='Image prediction model')
	
	parser.add_argument('-lr', default=0.001, help='Learning rate', type=float)
	parser.add_argument('-lrd', default=0.00001 , help='Learning rate decay', type=float)
	parser.add_argument('-max_epoch', default=6000, help='Max epoch', type=int)
	parser.add_argument('-save_interval', default=300, help='Model saving interval in epochs', type=int)

	args = parser.parse_args()

	torch.cuda.set_device(0)
	
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

	if args.image_model == 'Simple':
		image_model = ImageModelSimple().cuda()
		rand_input = torch.FloatTensor(1,1,16,16).normal_().cuda()
	elif args.image_model == 'ResNet':
		image_model = ResNet(num_input_channels=1, num_output_channels=1, num_blocks=18, num_channels=2, need_sigmoid=False).cuda()
		rand_input = torch.FloatTensor(1,1,86,86).normal_().cuda()
	elif args.image_model == 'DCGAN':
		image_model = dcgan(inp=1, ndf=8, num_ups=4).cuda()
		rand_input = torch.FloatTensor(1,1,19,19).normal_().cuda()
	else:
		raise(Exception("unknown image model", args.image_model))

	loss_model = nn.MSELoss()

	trainer = PriorTrainer(	image_model = image_model,
							loss_model = loss_model,
							lr=float(args.lr), 
							lr_decay = float(args.lrd))
		
	trainer.new_log(os.path.join(EXP_DIR,"training_loss.dat"), log_dir=EXP_DIR)
	
	
	
	data_path = os.path.join(DATA_DIR, args.image_name)
	if not os.path.exists(data_path):
		raise(Exception("dataset not found", data_path))
	target_image = get_image(data_path)
	

	for epoch in tqdm(xrange(args.max_epoch)):
		loss = trainer.optimize(rand_input, target_image)
		if (epoch+1)%args.save_interval==0:
			trainer.save_image(rand_input, 'image_%d.th'%epoch)
		