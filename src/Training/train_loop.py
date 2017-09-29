import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import src
import matplotlib.pylab as plt
def train_loop(     train_dataset_stream,
					validation_dataset_stream,
					net,
					loss,
					cuda_dev = None,
					learning_rate = 0.001,
					start_epoch = None,
					max_epoch = 100,
					batch_size = 10,
					model_save_period = 10,
					model_save_dir = None,
					logger = None,
					test_visualization = True
		):
	net = net.cuda()
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
			
	if start_epoch is None:
		start_epoch = 0
	

	total_error_train = []
	total_error_val = []
	for epoch in range(start_epoch, max_epoch):
		#Training step
		net.train()
		error_training = []
		for n,data in tqdm(enumerate(train_dataset_stream), total=len(train_dataset_stream)):
			paths, x, y = data
			x, y = Variable(x.cuda()), Variable(y.cuda())
		
			optimizer.zero_grad()
			net_output = net(x)

			more_error = .0001*torch.abs(net_output.sum() - y.sum())

			error = loss(net_output, y) + more_error
						
			error.backward()
			optimizer.step()

			error_training.append(error.data[0])

			if n==0 and ((epoch+1) % model_save_period==0):
				a = torch.FloatTensor(64,64)
				a.copy_(net_output.data[0,:,:])
				ax = plt.subplot(2,2,1)
				ax.imshow(a.numpy())
				
				ax = plt.subplot(2,2,2)
				a.copy_(y.data[0,:,:])
				ax.imshow(a.numpy())

				ax = plt.subplot(2,2,3)
				a.copy_(x.data[0,0,0,:,:])
				ax.imshow(a.numpy())
				
				plt.savefig(os.path.join(logger, 'training_%d.png'%epoch))
						
		#Validation step
		if not (validation_dataset_stream is None):
			net.eval()
			error_validation = []
			for n, data in tqdm(enumerate(validation_dataset_stream), total=len(validation_dataset_stream)):
				paths, x, y = data
				x, y = Variable(x.cuda()), Variable(y.cuda())

				net_output = net(x)
				error = loss(net_output, y)

				error_validation.append(error.data[0])

				if n==0 and ((epoch+1) % model_save_period==0):
					a = torch.FloatTensor(64,64)
					a.copy_(net_output.data[0,:,:])
					ax = plt.subplot(2,2,1)
					ax.imshow(a.numpy())
					
					ax = plt.subplot(2,2,2)
					a.copy_(y.data[0,:,:])
					ax.imshow(a.numpy())

					ax = plt.subplot(2,2,3)
					a.copy_(x.data[0,0,0,:,:])
					ax.imshow(a.numpy())
					
					plt.savefig(os.path.join(logger, 'validation_%d.png'%epoch))
				
		print 'Training epoch %d ended'%epoch
		print 'Training error', np.average(error_training)
		print 'Validation error', np.average(error_validation)
		total_error_train.append(np.average(error_training))
		total_error_val.append(np.average(error_validation))
		if (epoch+1) % model_save_period == 0:
			plt.figure()
			plt.plot(total_error_train, label = 'train')
			plt.plot(total_error_val, label = 'validation')
			plt.legend()
			plt.savefig(os.path.join(logger, 'progress.png'))

		#Saving the model
		if not(model_save_dir is None):
			if (epoch+1) % model_save_period == 0:
				torch.save(net.state_dict(), os.path.join(model_save_dir, 'net_epoch_%d.pth'%(epoch+1)))
				
