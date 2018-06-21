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

class PriorTrainer:
	def __init__(	self, image_model, loss_model, lr=0.0001, lr_decay=0.0001):
		self.lr = lr
		self.lr_decay = lr_decay
		self.image_model = image_model
		self.loss_model = loss_model
		self.optimizer = optim.Adam(self.image_model.parameters(), lr=self.lr)
		self.log = None
		self.log_dir = None
		self.lr_scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0/(1.0+epoch*self.lr_decay))

		print 'Optimizer: ADAM'
		print 'Learning rate = ', self.lr
		print 'Learning rate decay = ', self.lr_decay
				
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
	def optimize(self, input, image):
		"""
		Optimization step. 
		Input: input, target image
		Output: loss
		"""
		self.image_model.train()
		self.loss_model.train()
		self.optimizer.zero_grad()
		x = Variable(input.cuda())
		y = Variable(image.cuda())
				
		pred_image = self.image_model(x)
		L = self.loss_model(pred_image, y)
		
		L.backward()
		
				
		if not self.log is None:
			self.log.write("Loss\t%f\n"%(L.data[0]))
			
		self.optimizer.step()
		self.lr_scheduler.step()
		return L.data[0]

	def save_image(self, input, filename):
		self.image_model.eval()
		x = Variable(input.cuda())
		pred_image = self.image_model(x)
		torch.save(pred_image.data.cpu(), os.path.join(self.log_dir, filename))
	
	def get_model_filename(self):
		return "predictor"

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