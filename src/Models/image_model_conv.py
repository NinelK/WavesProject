import os
import sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import math
from torch.utils.serialization import load_lua

class ImageModelSimple(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(ImageModelSimple, self).__init__()
		
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(1, 4, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.BatchNorm3d(4),
			nn.ReLU(),
			nn.ConvTranspose2d(4, 4, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm3d(4),
			nn.ReLU(),
			nn.ConvTranspose2d(4, 1, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.ReLU(),
			nn.Upsample(size=(86,86),mode='bilinear')
		)

	def forward(self, input):
		deconv_out = self.deconv(input)
		
		return deconv_out