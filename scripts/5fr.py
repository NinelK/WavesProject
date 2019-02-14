import _pickle as pkl
import numpy as np
from PIL import Image
import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Process frames into 5frame pkls.')
parser.add_argument('-dir', default='', help='Directory with frames')
parser.add_argument('-name', default='', help='Output file names')

args = parser.parse_args()

path = "../dataset"

os.chdir(path)

N = len(glob.glob1(args.dir,"*.png")) - 4 # -5frames + 1

if N>0:
	for j in range(N):
		list = []
		for i in range(5):
			path = "./%s/Frame%03d.png" % (args.dir,i+j)
			with open(path, 'rb') as fin:
				data = np.asarray(Image.open(fin))
			list.append(data)

		mov = np.array(list)

		with open("./%s/%s%03d.pkl" % (args.dir,args.name,j), 'wb') as fin:
			pkl.dump({"in" : mov, "out" : mov},fin)

		os.system("ls ./%s/*.pkl > ./%s/video_set.dat" % (args.dir,args.dir))

os.system("cat ./video_*/video_set.dat > video_set.dat")
