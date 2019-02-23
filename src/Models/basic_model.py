import os
import sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import math
from torch.utils.serialization import load_lua

class BasicModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(BasicModel, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(num_input_channels, 16, (3,3,3), dilation = (1,3,3)),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.MaxPool3d( (3,3,3), (2,2,2) ),

			nn.Conv3d(16, 32, (1,3,3)),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.MaxPool3d((1,3,3), (1,2,2))
		)

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.ReLU(),
			nn.Upsample(size=(86,86),mode='bilinear')
		)

	def forward(self, input):
		conv_out = self.conv(input)
		conv_out = conv_out.squeeze()
		#print(conv_out.size())
		deconv_out = self.deconv(conv_out)
		return deconv_out.squeeze()
