import os
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F

class VAELossModel(nn.Module):
	def __init__(self, num_input_channels = 1):
		super(VAELossModel, self).__init__()
		self.BCE = torch.nn.BCELoss(size_average=True)

	def forward(self, recon_x, x, mu, logsigma):
		BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
		# BCE = self.BCE(recon_x, x)
		KLD = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
		# print(BCE)
		# print(KLD)
		return BCE + KLD
		