import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import atexit
import numpy as np
import math
import os
import sys
from matplotlib import pylab as plt
from torch.optim.lr_scheduler import LambdaLR

class CentresTrainer:
	def __init__(	self, image_model, loss_model, lr=0.0001):
		self.lr = lr
		self.image_model = image_model
		self.loss_model = loss_model
		self.optimizer = optim.Adam(self.image_model.parameters(), lr=self.lr)
		self.log = None
		self.log_dir = None
		
		print('Optimizer: ADAM')
		print('Learning rate = ', self.lr)
						
		atexit.register(self.cleanup)
	
	def new_log(self, log_file_name, log_dir=None):
		if not self.log is None:
			self.log.close()
		self.log = open(log_file_name, "w")

		if not log_dir is None:
			self.log_dir = log_dir
			if not os.path.exists(self.log_dir):
				os.mkdir(self.log_dir)

	def cleanup(self):
		if not self.log is None:
			self.log.close()
		
	# @profile
	def optimize(self, data, mask):
		"""
		Optimization step. 
		Input: data
		Output: loss
		"""
		self.image_model.train()
		self.loss_model.train()
		self.optimizer.zero_grad()
		_, x, y = data
		x = x
		y = y
		x = Variable(x.cuda())				
		y = Variable(y.cuda())
		
		pred = self.image_model(x) * mask

		more_error = 0.000001*torch.abs(pred.sum() - y.sum())

		L = self.loss_model(pred, y) + more_error
		
		L.backward()
						
		if not self.log is None:
			self.log.write("Loss\t%f\n"%(L.data))
			
		self.optimizer.step()
		return L.data

	def predict(self, data, mask):
		"""
		Prediction step. 
		Input: data
		Output: loss
		"""
		self.image_model.eval()
		self.loss_model.eval()
		path, x, y = data
		x = Variable(x.cuda())
		y = Variable(y.cuda())
		
		pred = self.image_model(x) * mask
		L = self.loss_model(pred, y)
				
		if not self.log is None:
			self.log.write("Loss\t%f\n"%(L.data))

		if not self.log_dir is None:
			for i in range(pred.size(0)):
				name = path[i].split('.')[0].split('/')[-1]
				torch.save(pred[i,:,:].cpu(), os.path.join(self.log_dir, name+'_pred.th'))
				torch.save(y[i,:,:].cpu(), os.path.join(self.log_dir, name+'_grnd.th'))
				torch.save(x[i,0,2,:,:].cpu(), os.path.join(self.log_dir, name+'_sign.th'))				
				#import matplotlib.pylab as plt
				#fig = plt.figure()
				#ax = plt.axes()
				#f = torch.cat([x[i,0,2,:,:].cpu(),pred[i,2,:,:].cpu()], dim=0).data.numpy()
				#image = ax.imshow(f)
				# plt.savefig(filename)
				#plt.show()
			
		return L.data

		
	def get_model_filename(self):
		return "Centre"

	def save_models(self, epoch, directory):
		"""
		saves the model
		"""
		torch.save(self.image_model.state_dict(), os.path.join(directory, self.get_model_filename()+'_epoch%d.th'%epoch))
		torch.save(self.optimizer.state_dict(), os.path.join(directory, self.get_model_filename()+'_optim_epoch%d.th'%epoch))
		print('Model saved successfully')

	def load_models(self, epoch, directory):
		"""
		loads the model
		"""
		self.image_model.load_state_dict(torch.load(os.path.join(directory, self.get_model_filename()+'_epoch%d.th'%epoch)))
		self.optimizer.load_state_dict(torch.load(os.path.join(directory, self.get_model_filename()+'_optim_epoch%d.th'%epoch)))
		print('Model loaded succesfully')
