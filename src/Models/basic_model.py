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
			nn.Conv3d(num_input_channels, 8, (1,3,3) ),
			nn.BatchNorm3d(8),
			nn.ReLU(),
			nn.MaxPool3d( (1,3,3), (1,2,2) ),

			nn.Conv3d(8, 16, (1,3,3)),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.MaxPool3d((1,3,3), (1,2,2)),

			nn.Conv3d(16, 32, (3,3,3)),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.MaxPool3d((3,3,3), (2,2,2)),

			nn.Conv3d(32, 64, (3,3,3)),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.MaxPool3d((2,3,3), (1,2,2)),
		)

		self.conv2d = nn.Sequential(

			#nn.Conv2d(32,64,(3,3)),
			#nn.BatchNorm2d(64),
			#nn.ReLU(),
			#nn.MaxPool2d((3,3),(2,2)),

			#nn.Conv2d(64,128,(3,3)),
			#nn.BatchNorm2d(128),
			#nn.ReLU(),
			#nn.MaxPool2d((3,3),(2,2))
		)

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 1, padding = 0, bias = False),
			nn.BatchNorm2d(16),
			nn.ReLU(),

			nn.ConvTranspose2d(16, 4, kernel_size = 3, stride =1, padding = 0, bias=False),
			nn.BatchNorm2d(4),
			nn.ReLU(),

			nn.ConvTranspose2d(4, 1, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.ReLU(),
			nn.Upsample(size=(86,86),mode='bilinear')
		)

	def forward(self, input):
		conv_out = self.conv(input)
		conv_out = conv_out.squeeze()
		#conv_out2 = self.conv2d(conv_out)
		deconv_out = self.deconv(conv_out)
		return deconv_out.squeeze()
