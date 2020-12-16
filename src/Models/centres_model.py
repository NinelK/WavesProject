import os 
import sys 
import torch.nn as nn 
import torch

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class CentresModel(nn.Module):

	def __init__(self, num_input_channels = 4):
		super(CentresModel, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv3d(num_input_channels, 16, kernel_size = (3,3,3), dilation = (1,3,3)),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.MaxPool3d((3,3,3), (1,2,2)),

			nn.Conv3d(16, 32, kernel_size = (1,3,3), dilation = (1,3,3)),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.MaxPool3d((1,3,3), (1,2,2))

		)

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 1, padding = 0, bias=True),
			nn.BatchNorm2d(16),
			nn.ReLU(),

			nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 1, padding = 0, bias=True),
			nn.ReLU(),

			Interpolate(size=(86, 86),mode='bilinear')
		)

	def forward(self, input):

		conv_out = self.conv(input)
		deconv_out = self.deconv(conv_out.squeeze(-3))

		return deconv_out.squeeze()
