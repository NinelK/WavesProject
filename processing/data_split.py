import glob
import numpy as np
from scipy import misc
from tqdm import tqdm, trange

singFiles 	= glob.glob("./answers/sing*")
pkls 		= glob.glob("./pkls/*.pkl")
pkls = [name.split('/')[-1] for name in pkls]

training_list_path = "./pkls/training.csv"
validation_list_path = "./pkls/validation.csv"

img = np.zeros(shape=(86,86), dtype=np.float32)

mask = np.zeros(shape=(86,86), dtype=np.int8)
mask[:28,:] = 1									#first 28 px of the image = 20% of the circle (60px in diameter in 86x86px image)

with open(training_list_path, 'w') as ftrain, open(validation_list_path, 'w') as fvalid:
	for sing,pkl in tqdm(zip(singFiles,pkls)):
		read = misc.imread(sing)
		img+= read
		if np.sum(read*mask) == 0:
			ftrain.write(pkl+'\n')
		else:
			fvalid.write(pkl+'\n')