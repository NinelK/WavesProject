import os 
import sys 
import torch.nn as nn 
import torch

class VAEModel(nn.Module):

	def __init__(self, num_input_channels = 4):
		super(VAEModel, self).__init__()

		self.max_ch = 64
		self.pic_size = 9
		self.h_size = 1024
		self.k = 2

		self.conv = nn.Sequential(
			nn.Conv3d(num_input_channels, int(self.max_ch/self.k**2), kernel_size = (3,4,4), stride = (1,2,2)),
			nn.ReLU(),

			nn.Conv3d(int(self.max_ch/self.k**2), int(self.max_ch/self.k), kernel_size = (3,4,4), stride = (1,2,2)),
			nn.ReLU(),

			nn.Conv3d(int(self.max_ch/self.k), self.max_ch, kernel_size = (1,4,4), stride = (1,2,2)),
			nn.ReLU(),

		)

		self.fc_encode_mu = nn.Sequential(nn.Linear(self.max_ch*self.pic_size**2, self.h_size), nn.ReLU())
		self.fc_encode_sigma = nn.Sequential(nn.Linear(self.max_ch*self.pic_size**2, self.h_size), nn.ReLU())

		self.fc_decode = nn.Sequential(nn.Linear(self.h_size, self.max_ch*self.pic_size**2), nn.ReLU())

		self.deconv = nn.Sequential(
			nn.ConvTranspose3d(self.max_ch, int(self.max_ch/self.k), kernel_size = (1,4,4), stride = (1,2,2), padding = 0, bias=True),
			nn.ReLU(),

			nn.ConvTranspose3d(int(self.max_ch/self.k), int(self.max_ch/self.k**2), kernel_size = (3,4,4), stride = (1,2,2), padding = 0, bias=True),
			nn.ReLU(),

			nn.ConvTranspose3d(int(self.max_ch/self.k**2), 1, kernel_size = (3,4,4), stride = (1,2,2), padding = 0, bias=True),
			nn.Sigmoid(),
			
			#nn.UpsamplingBilinear3d(size=(5, 86, 86))
		)

	def encode(self, input):
		#input = input.unsqueeze(dim=1)
		conv_out = self.conv(input)
		#print(conv_out.size())
		conv_out = conv_out.view(-1,1,self.max_ch*self.pic_size**2)#conv_out.squeeze()
		#logsigma = self.fc_encode_sigma(conv_out)
		mu = self.fc_encode_mu(conv_out)
		return mu #, logsigma

	def decode(self, input):
		x = self.fc_decode(input)
		#print(x.size())
		x=x.view(-1,self.max_ch,1,self.pic_size,self.pic_size)#x = x.unsqueeze(dim=2).unsqueeze(dim=3)
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
