import os 
import sys 
import torch.nn as nn 
import torch

class CentresModel(nn.Module):

	def __init__(self, num_input_channels = 4):
		super(CentresModel, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv3d(num_input_channels, 16, kernel_size = (5,3,3)),
			nn.BatchNorm3d(16),
			nn.ReLU(),
		)
		

		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size = (3,3)),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = (3,3)),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)

		self.deconv1 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 0, bias=True),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)


		self.deconv2 = nn.Sequential(
			nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 1, padding = 0, bias=True),
			nn.BatchNorm2d(16),
			nn.ReLU()
		)

		self.deconv3 = nn.Sequential(
			nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 1, padding = 0, bias=True),
			nn.ReLU()
		)
		
		self.pool = nn.MaxPool2d(3,stride=2, return_indices = True)
		self.unpool = nn.MaxUnpool2d(3,stride=2)
		
		self.upsample = nn.Upsample(size=(86, 86),mode='bilinear')

			
	def forward(self, input):
		
		conv_out1 = self.conv1(input)

		conv_out1 = conv_out1.squeeze()

		pool1,ind1 = self.pool(conv_out1)

		conv_out2 = self.conv2(pool1)

		pool2,ind2 = self.pool(conv_out2)
	
#		print(conv_out.size())
		out = self.conv3(pool2)
	
		deconv_out1 = self.deconv1(out)
		unpool2 = self.unpool(deconv_out1,ind2)		
		deconv_out2 = self.deconv2(unpool2)
		unpool1 = self.unpool(deconv_out2,ind1)
		deconv_out3 = self.deconv3(unpool1)

		deconv_out = self.upsample(deconv_out3)

		#print(deconv_out.size())
		return deconv_out.squeeze()
