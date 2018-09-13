import os
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import math
from torch.utils.serialization import load_lua

class CenterModel(nn.Module):
	def __init__(self, num_input_channels = 1, batch_size=32):
		super(CenterModel, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size = 3, stride = 3, padding = 0, bias=False),

			nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size = 3, stride = 3, padding = 0, bias=False),

			nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size = 4, stride = 4, padding = 0, bias=False),
		).cuda()

		self.fc = nn.Sequential(
				nn.Linear(64, 32),
				nn.ReLU(),
				nn.Linear(32, 8),
				nn.Sigmoid(),
				nn.Linear(8, 2),
				nn.Sigmoid()
				).cuda()

		
		view_size = 86
		
		self.x_span = Variable(torch.FloatTensor(batch_size, view_size, view_size).fill_(0).cuda())
		self.y_span = Variable(torch.FloatTensor(batch_size, view_size, view_size).fill_(0).cuda())

		for i in xrange(batch_size):
			for j in xrange(view_size):
				self.x_span.data[i,:,j].copy_(torch.linspace(0.0, 1.0, view_size))
				self.y_span.data[i,j,:].copy_(torch.linspace(0.0, 1.0, view_size))
		
		self.x_span = torch.unsqueeze(self.x_span, dim=1)
		self.y_span = torch.unsqueeze(self.y_span, dim=1)


		

	def forward(self, input):
		batch_size = input.size(0)

		input = input.unsqueeze(dim=1)
		input = torch.cat([input, self.x_span[:batch_size,:,:,:], self.y_span[:batch_size,:,:,:]], dim=1)
		
		conv_out = self.conv(input)
		
		conv_out = conv_out.squeeze()
		conv_out.size()
		fc_out = self.fc(conv_out)
		
		return fc_out