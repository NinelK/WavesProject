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

		self.lstm_encode = torch.nn.LSTM(	input_size = 64, 
									hidden_size = 16, 
									num_layers = 2, 
									batch_first = True,
									bidirectional = False)

		self.fc_encode_mu = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
		self.fc_encode_sigma = nn.Sequential(nn.Linear(16, 8), nn.ReLU())

		self.lstm_decode = torch.nn.LSTM(	input_size = 8, 
											hidden_size = 64, 
											num_layers = 2, 
											batch_first = True,
											bidirectional = False)

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(1, 1, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.BatchNorm3d(1),
			nn.ReLU(),

			nn.ConvTranspose2d(1, 1, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.BatchNorm3d(1),
			nn.ReLU(),

			nn.ConvTranspose2d(1, 1, kernel_size = 3, stride = 2, padding = 0, bias=False),
			nn.BatchNorm3d(1),
			nn.Sigmoid(),
			
			nn.Upsample(size=(86, 86),mode='bilinear')
		)

	def encode(self, input):
		batch_size = input.size(0)
		num_frames = input.size(1)
		im_size = input.size(2)
		x = input.resize(batch_size*num_frames, 1, im_size, im_size).contiguous()
		

		conv_out = self.conv(x)
		num_features = conv_out.size(1)
		lstm_in = conv_out.resize(batch_size, num_frames, num_features).contiguous()
		

		lstm_out, _ = self.lstm_encode(lstm_in)
		num_features_lstm = lstm_out.size(2)
		encoding = lstm_out.resize(batch_size*num_frames, num_features_lstm)

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

		lstm_output, _ = self.lstm_decode(input)
		lstm_output = lstm_output.resize(batch_size*num_frames, 1, 8, 8).contiguous()
		

		deconv_out = self.deconv(lstm_output)
		y = deconv_out.resize(batch_size, num_frames, 86, 86)

		return y

	def reparameterize(self, mu, sigma):
		if self.training:
			std = torch.exp(0.5*sigma)
			z = torch.normal(mu, std)
			return z
		else:
			return mu

	def forward(self, input):
		mu, sigma = self.encode(input)
		z = self.reparameterize(mu, sigma)
		y = self.decode(z)

		return y, mu, sigma
