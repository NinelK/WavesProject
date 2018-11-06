import os
import sys
import torch.nn as nn
import torch


class VAEModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(VAEModel, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(num_input_channels, 8, (3,3) ),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.MaxPool2d( (3,3), (2,2) ),

			nn.Conv2d(8, 16, (3,3)),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d((3,3), (3,3)),

			nn.Conv2d(16, 32, (3,3)),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d((3,3), (3,3)),

			nn.Conv2d(32, 64, (3,3)),
			nn.BatchNorm2d(64),
			nn.ReLU(),
		)

		self.fc_encode_mu = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
		self.fc_encode_sigma = nn.Sequential(nn.Linear(64, 256), nn.ReLU())

		self.fc_decode = nn.Sequential(nn.Linear(256, 64), nn.ReLU())

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.BatchNorm3d(32),
			nn.ReLU(),

			nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.BatchNorm3d(16),
			nn.ReLU(),

			nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.Sigmoid(),
			
			nn.Upsample(size=(86, 86),mode='bilinear')
		)

	def encode(self, input):
		input = input.unsqueeze(dim=1)
		conv_out = self.conv(input)
		conv_out = conv_out.squeeze()
		logsigma = self.fc_encode_sigma(conv_out)
		mu = self.fc_encode_mu(conv_out)
		return mu, logsigma

	def decode(self, input):
		x = self.fc_decode(input)
		x = x.unsqueeze(dim=2).unsqueeze(dim=3)
		deconv_out = self.deconv(x)
		return deconv_out

	def reparameterize(self, mu, logsigma):
		if self.training:
			std = torch.exp(0.5*logsigma)
			z = torch.normal(mu, std)
			return z
		else:
			return mu

	def forward(self, input):
		
		mu, logsigma = self.encode(input)
		
		z = self.reparameterize(mu, logsigma)
		
		y = self.decode(z)
		y = y.squeeze()

		return y, mu, logsigma