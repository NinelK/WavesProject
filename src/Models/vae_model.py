import os
import sys
import torch.nn as nn
import torch
from torch.autograd import Variable


class VAEModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(VAEModel, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(num_input_channels, 8, kernel_size = 4),
			#nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.MaxPool2d((3,3), (2,2)),

			nn.Conv2d(8, 16, kernel_size = 3),
			#nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d((3,3), (2,2)),

			nn.Conv2d(16, 32, kernel_size = 3),
			#nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d((3,3), (2,2)),

			nn.Conv2d(32, 64, kernel_size = 3),
			#nn.BatchNorm2d(64),
			nn.ReLU(),

		)

		self.fc_encode_mu = nn.Sequential(nn.Linear(64*36, 32), nn.ReLU())
		self.fc_encode_sigma = nn.Sequential(nn.Linear(64*36, 32), nn.ReLU())
		self.fc_decode = nn.Sequential(nn.Linear(32,64*36), nn.ReLU())

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 0, bias=False),
			#nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.ConvTranspose2d(32, 16, kernel_size = 5, stride = 2, padding = 0, bias=False),
			#nn.BatchNorm3d(16),
			nn.ReLU(),

			nn.ConvTranspose2d(16, 8, kernel_size = 6, stride = 1, padding = 0, bias=False),
			#nn.BatchNorm3d(8),
			nn.ReLU(),

			nn.ConvTranspose2d(8, 1, kernel_size = 6, stride = 1, padding = 0, bias=False),
			nn.Sigmoid(),
			
			nn.Upsample(size=(86, 86),mode='bilinear')
		)

	def encode(self, input):
		batch_size = input.size(0)
		num_frames = input.size(1)
		im_size = input.size(2)
		x = input.resize(batch_size*num_frames, 1, im_size, im_size)
		
		conv_out = self.conv(x)
		num_features = conv_out.size(1)

		encoding = conv_out.resize(batch_size*num_frames,num_features*36)

		sigma = self.fc_encode_sigma(encoding)
		mu = self.fc_encode_mu(encoding)

		num_features_encoding = sigma.size(1)
		
		sigma = sigma.resize(batch_size, num_frames, num_features_encoding)
		mu = mu.resize(batch_size, num_frames, num_features_encoding)

		return mu, sigma

	def decode(self, input):
		batch_size = input.size(0)
		num_frames = input.size(1)
		num_features = input.size(2)

		if(num_features!=64*36):
			print("Error")

		lstm_output = input.resize(batch_size*num_frames, 64, 6, 6)

		deconv_out = self.deconv(lstm_output)
		y = deconv_out.resize(batch_size, num_frames, 86, 86)

		return y

	def reparameterize(self, mu, sigma):
		batch_size = mu.size(0)
		num_frames = mu.size(1)
		num_features = mu.size(2)

		sigma = sigma.resize(batch_size*num_frames, num_features)
		mu = mu.resize(batch_size*num_frames, num_features)

		if self.training:
			std = torch.exp(0.5*sigma)
			#z = torch.normal(mu, std)
			esp = torch.randn(*mu.size())
			esp = esp.cuda()
        		#z = esp.mul(std).add_(mu)
			z = mu + std * Variable(esp)
		else:
			z = mu

		z = self.fc_decode(z)
		z = z.resize(batch_size,num_frames, 64*36)

		return z


	def forward(self, input):
		#mu = 0
		#sigma = 0
		mu, sigma = self.encode(input)
		z = self.reparameterize(mu, sigma)
		#batch_size = input.size(0)
		#num_frames = input.size(1)
		#im_size = input.size(2)
		#x = input.resize(batch_size*num_frames, 1, im_size, im_size)
		
		#conv_out = self.conv(x)
		#num_features = conv_out.size(1)

		#encoding = conv_out.resize(batch_size,num_frames,num_features*36)

		y = self.decode(z)

		return y, mu, sigma
