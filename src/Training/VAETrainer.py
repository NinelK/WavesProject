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

class VAETrainer:
	def __init__(	self, image_model, loss_model, lr=0.0001):
		self.lr = lr
		self.image_model = image_model
		self.loss_model = loss_model
		self.optimizer = optim.Adam(self.image_model.parameters(), lr=self.lr)
		self.log = None
		self.log_dir = None
		
		print 'Optimizer: ADAM'
		print 'Learning rate = ', self.lr
						
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
		_, x = data
		x = Variable(x.cuda())

		#W = 86
		#xx, yy = np.ogrid[:W,:W]
		#mask = torch.from_numpy(np.array((xx-(W-1)/2)**2 + (yy-(W-1)/2)**2 < (43)**2).astype("float32"))
		mask = Variable(mask.cuda())
						
		pred, mu, sigma = self.image_model(x)
		pred = pred * mask
		L = self.loss_model(pred, x, mu, sigma)
		
		L.backward()
						
		if not self.log is None:
			self.log.write("Loss\t%f\n"%(L.data[0]))
			
		self.optimizer.step()
		return L.data[0]

	def predict(self, data, mask):
		"""
		Prediction step. 
		Input: data
		Output: loss
		"""
		self.image_model.eval()
		self.loss_model.eval()
		path, x = data
		x = Variable(x.cuda())

		mask = Variable(mask.cuda())

		pred, mu, sigma = self.image_model(x)
		pred = pred * mask
		L = self.loss_model(pred, x, mu, sigma)
						
		if not self.log is None:
			self.log.write("Loss\t%f\n"%(L.data[0]))

		if not self.log_dir is None:
			for i in xrange(pred.size(0)):
				name = path[i].split('.')[0].split('/')[-1]
				print name
				torch.save(pred[i,:,:,:].cpu(), os.path.join(self.log_dir, name+'_pred.th'))
				torch.save(x[i,:,:,:].cpu(), os.path.join(self.log_dir, name+'_grnd.th'))
			
		return L.data[0]

		
	def get_model_filename(self):
		return "VAE"

	def save_models(self, epoch, directory):
		"""
		saves the model
		"""
		torch.save(self.image_model.state_dict(), os.path.join(directory, self.get_model_filename()+'_epoch%d.th'%epoch))
		torch.save(self.optimizer.state_dict(), os.path.join(directory, self.get_model_filename()+'_optim_epoch%d.th'%epoch))
		print 'Model saved successfully'

	def load_models(self, epoch, directory):
		"""
		loads the model
		"""
		self.image_model.load_state_dict(torch.load(os.path.join(directory, self.get_model_filename()+'_epoch%d.th'%epoch)))
		self.optimizer.load_state_dict(torch.load(os.path.join(directory, self.get_model_filename()+'_optim_epoch%d.th'%epoch)))
		print 'Model loaded succesfully'
