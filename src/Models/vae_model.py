import os
import sys
import torch.nn as nn
import torch

class VAEModel(nn.Module):
	def __init__(self, num_input_channels = 3):
		super(VAEModel, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(num_input_channels, 8, kernel_size = 4, stride = 2),
			nn.ReLU(),

			nn.Conv2d(8, 16, kernel_size=4, stride = 2),
			nn.ReLU(),
			
			nn.Conv2d(16, 32, kernel_size=4, stride = 2),
			nn.ReLU(),
	
			nn.Conv2d(32, 64, kernel_size=3, stride = 2),
			nn.ReLU(),
			
		)

		self.fc_encode_mu = nn.Sequential(nn.Linear(16*64, 1024), nn.ReLU())
		#self.fc_encode_sigma = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

		self.fc_decode = nn.Sequential(nn.Linear(1024, 64*16), nn.ReLU())

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size = 6, stride = 2, padding = 0, bias=True),
			nn.ReLU(),

			nn.ConvTranspose2d(32, 16, kernel_size = 5, stride = 2, padding = 0, bias=True),
			nn.ReLU(),

			nn.ConvTranspose2d(16, 8, kernel_size = 5, stride = 2, padding = 0, bias=True),
			nn.ReLU(),

			nn.ConvTranspose2d(8, 1, kernel_size = 5, stride = 1, padding = 0, bias=True),
			nn.Sigmoid(),
			
			nn.UpsamplingBilinear2d(size=(86, 86))
		)

	def encode(self, input):
		#input = input.unsqueeze(dim=1)
		conv_out = self.conv(input)
		#print(conv_out.size())
		conv_out = conv_out.view(-1,64*16)#conv_out.squeeze()
		#print(conv_out.size())
		#logsigma = self.fc_encode_sigma(conv_out)
		mu = self.fc_encode_mu(conv_out)
		return mu#, logsigma

	def decode(self, input):
		x = self.fc_decode(input)
		#print(x.size())
		x=x.view(-1,64,4,4)#x = x.unsqueeze(dim=2).unsqueeze(dim=3)
		#print(x.size())
		deconv_out = self.deconv(x)
		# print(deconv_out.size())
		return deconv_out

	def reparameterize(self, mu, logsigma):
		if self.training:
			std = torch.exp(0.5*logsigma)
			esp = torch.randn(*mu.size()).cuda()
			z = mu + std*esp
			return z
		else:
			return mu

	def forward(self, input):
		
		mu = self.encode(input)

		#z = self.reparameterize(mu, logsigma)
		
		y = self.decode(mu)
		y = y.squeeze()

		return y#, mu, logsigma
