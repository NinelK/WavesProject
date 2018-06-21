import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src import DATA_DIR
import numpy as np

def get_image(filename):
	I = plt.imread(filename).astype(dtype='float32')
	I = (I - np.mean(I))/np.std(I)
	image = torch.from_numpy(I)
	image = image.unsqueeze(dim=0).unsqueeze(dim=0)
	return image.cuda()


if __name__=='__main__':
	tiff_file = '../../dataset/SpiralsInPetri/Spiral1_circ.tif'

	I = get_image(tiff_file)
	plt.imshow(I[0,0,:,:].cpu().numpy())
	plt.show()

	I = np.array(I).flatten()
	plt.hist(I)
	plt.show()
