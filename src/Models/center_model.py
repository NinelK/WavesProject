import os
import sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import math
from torch.utils.serialization import load_lua

class CenterModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(CenterModel, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(4),

			# nn.Conv2d(128, 256, kernel_size = 3, stride = 0, padding = 0, bias=False),
			# nn.BatchNorm2d(256),
			# nn.ReLU(),
		)

		self.fc = nn.Sequential(
				nn.Linear(128, 32),
				nn.ReLU(),
				nn.Linear(32, 8),
				nn.Sigmoid(),
				nn.Linear(8, 2),
				nn.Sigmoid()
				)

	def forward(self, input):
		
		input = input.unsqueeze(dim=1)
		conv_out = self.conv(input)
		
		conv_out = conv_out.squeeze()
		conv_out.size()
		fc_out = self.fc(conv_out)
		
		return fc_out