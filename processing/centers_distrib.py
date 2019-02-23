import glob
import numpy as np
from scipy import misc
from tqdm import tqdm, trange

singFiles = glob.glob("./answers/sing*")
centFiles = glob.glob("./answers/cent*")

img = np.zeros(shape=(86,86), dtype=np.float32)

for file in tqdm(singFiles):
	read = misc.imread(file)
	img+= read

misc.imsave("hotmap_sing.png",img)

for file in tqdm(centFiles):
	read = misc.imread(file)
	img+= read

misc.imsave("hotmap_cent.png",img)