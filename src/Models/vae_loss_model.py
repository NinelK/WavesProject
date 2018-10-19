import os
import sys
import torch.nn as nn
import torch


class VAELossModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(VAELossModel, self).__init__()
		self.BCE = torch.nn.BCELoss(size_average=False)
		self.L1 = torch.nn.L1Loss()

	def forward(self, recon_x, x, mu, sigma):
		#fr = x.size(0)
		BCE = self.BCE(recon_x, x)/(10.0*86*86)
		#L1 = self.L1(recon_x,x)
		#sigma2 = sigma.pow(2)
		KLD = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())/32.0
		#more_error = 0.001*(torch.abs(recon_x.max() - x.max())+torch.abs(recon_x.min() - x.min()))
		#KLD /= 86*86

		return BCE + KLD
		
