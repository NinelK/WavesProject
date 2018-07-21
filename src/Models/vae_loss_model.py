import os
import sys
import torch.nn as nn
import torch


class VAELossModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(VAELossModel, self).__init__()
		self.BCE = torch.nn.BCELoss(size_average=True)
		#self.L1 = torch.nn.L1Loss()

	def forward(self, recon_x, x, mu, sigma):
		BCE = self.BCE(recon_x, x)
		#L1 = self.L1(recon_x,x)
		KLD = torch.abs(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()))
		
		return BCE + KLD
		
