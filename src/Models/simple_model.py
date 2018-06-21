import os
import sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import math
from torch.utils.serialization import load_lua

class SimpleModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(SimpleModel, self).__init__()
		self.conv3d = nn.Sequential(
			nn.Conv3d(num_input_channels, 16, (5,5,5) ),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.MaxPool3d( (6,3,3), (2,2,2)),
		)
		self.conv2d = nn.Sequential(
			nn.Conv2d(16, 1, (3,3) ),
			# nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.MaxPool2d( (3,3), (2,2)),

		)
		# self.deconv = nn.Sequential(
		# 	nn.ConvTranspose2d(8, 1, kernel_size = 7, stride = 1, padding = 0, bias=False),
		# 	nn.ReLU(),
		# 	nn.MaxPool2d( (5,5), (2,2)),
		# 	# nn.Upsample(size=(64,64),mode='bilinear')
		# )
		# self.pad = nn.ZeroPad2d((2, 2, 2, 2))
		self.upsample = nn.Upsample(size=(64,64), mode='bilinear')

	def forward(self, input):
		conv_out = self.conv3d(input)
		conv_out = conv_out.squeeze()
		# print conv_out.size()
		conv_out = self.conv2d(conv_out)
		# deconv_out = self.deconv(conv_out)
		
		deconv_out = self.upsample(conv_out)
		return deconv_out.squeeze()