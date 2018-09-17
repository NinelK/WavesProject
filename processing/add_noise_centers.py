from __future__ import print_function

import numpy as np
from scipy import misc
import scipy
import scipy.ndimage
import scipy.signal 
import os
import cPickle as pkl
from tqdm import tqdm, trange
#import matplotlib.pyplot as plt

T = 120						#duration of the raw data clips
dur = 11					#duration of the output
Ts = 88						#starting frame, where spiral is already formed&stabilised for sure
sobel_threshold = 50/4		#threshold for a "sharp" gradient
#front_threshold = 50/4		#another (similar) threshold for outlining the fronts
sgma = 3					#gaussian filter size used prior to 
sigma_MAX = 10.0
sigma_min = 3.0

#circular mask for a big image
Wb = 256
xx, yy = np.ogrid[:Wb,:Wb]
sel = (xx-(Wb-1)/2)**2 + (yy-(Wb-1)/2)**2 > (125)**2

#circular mask for a small image
W = 86
WF = 86.0
xx, yy = np.ogrid[:W,:W]
selF = np.uint8( (xx-(WF-1)/2)**2 + (yy-(WF-1)/2)**2 < (WF/2)**2 )

#parameters for signal distortion
po = 0.1	#percentage out of range
sgmaN = 5	#gaussian filter size

#function that flips and rotates images; flipX/Y/trans = [0,1]
def distort(img,flipX,flipY,trans):
		return img[::(flipX*2-1),::(flipY*2-1)].transpose(trans,1-trans)

gradx = lambda img: scipy.ndimage.sobel(img, axis=0, mode='nearest')
grady = lambda img: scipy.ndimage.sobel(img, axis=1, mode='nearest') 

#signal normalization
def normal(sig):
	if np.max(sig)!=np.min(sig):
		return (sig - np.min(sig))/(np.max(sig) - np.min(sig))
	else:
		return sig

def find_singularities(image,growing):
	Gx = gradx(image)
	Gy = grady(image)

	gm = np.sqrt(Gx**2 + Gy**2)
	under_selection = (gm < sobel_threshold) | ~growing		#sharp gradients indicate edges, and the wavefront is the one where signal grows over time
	#front_selection = gm > front_threshold
	gm[under_selection] = 1.0
	gx, gy = Gx/gm, Gy/gm
	gx[under_selection] = 0
	gy[under_selection] = 0

	rotor = -gradx(gy) + grady(gx)

	rotorB = scipy.ndimage.gaussian_filter(rotor, sigma=4)
	max_rotor = rotorB[~np.isnan(rotorB)].max()
	mask = (rotorB == max_rotor).astype('float32')
	(x, y) = np.where(mask == 1)
	return ([x[0],y[0]])

for runN in tqdm(range(120)):

	list = []	# list of centre coordinates

	imagesC = np.zeros(shape=(T+1,Wb,Wb), dtype=np.float32)	#for centre detection
	imagesS = np.zeros(shape=(T+1,W,W), dtype=np.float32)	#for noise adding, already small size

	# LOAD DATA
	
	for i in range(T+1):
		read = misc.imread("./sample%d/Movie%04d.png" % (runN, Ts + i)).astype("float32")
		imagesC[i] = scipy.ndimage.gaussian_filter(read, sigma=sgma)
		imagesS[i] = misc.imresize(read[3:-3,3:-3],(W,W))

	# FIND CENTRES

	for i in range(T):
		growing = imagesC[i+1] > imagesC[i]
		image = imagesC[i]
		image[sel] = np.nan
		coords = find_singularities(image,growing)
		#misc.imsave("./answers/gm_%03d.png" % (i),gm)
		list.append(coords)

	coords = np.array(list)

	b, a = scipy.signal.butter(4, 0.3, analog=False)
	coordX = scipy.signal.filtfilt(b, a, coords[:,0])
	coordY = scipy.signal.filtfilt(b, a, coords[:,1])

	#ADD NOISE

	amp = np.mean(imagesS[:,21:64,21:64])/2
	ampP = amp/2

	img = np.zeros(shape=(W,W,3), dtype=np.uint8)

	#for i in range(T):
	#	noise = np.random.normal(0, 1, size=imagesS[i].shape)*amp - amp/2
	#	spnoise = scipy.ndimage.gaussian_filter(noise, sigma=sgmaN)+scipy.ndimage.gaussian_filter(imagesS[i]+noise, sigma=3)
	#	noiseP = np.random.normal(0, 1, size=(W,W))*ampP - ampP/2
	#	signal = (spnoise+noiseP)
	#	min = np.min(signal)
	#	max = np.max(signal)
	#	signal[signal < (1-po) * min + po * max] = min
	#	signal[signal > po * min + (1-po) * max] = max
	#	imagesS[i] = signal*selF
		
	#for i in range(W):
	#	for j in range(W):
	#		signal = imagesS[:,i,j]
	#		imagesS[:,i,j] = normal(signal)

	#MAKE CLIPS

	listS = []	#list of successfull/unsuccessfull attempts to find centres

	for j in range(int(T/dur)):

		sing = np.zeros(shape=(W,W), dtype=np.float32)

		#recalculate center position in a smaller image
		cX = (coordX[j*dur:(j+1)*dur]-3) * 86/250
		cY = (coordY[j*dur:(j+1)*dur]-3) * 86/250
		#print((int(np.mean(cX)),int(np.mean(cY))))

		#make one-hot image
		sing[int(np.mean(cX)),int(np.mean(cY))] = 1
		#choose the maximum deviation of the center over the duration of a short (dur) clip
		sigma_center = np.max((np.std(cX),np.std(cY),sigma_min))

		if sigma_center < sigma_MAX:						#if deviation is not enormously big, the center was most likely detected correctly
			listS.append(1)
			#print(sigma_center)
			#apply gaussian filter to one-hot image and renormalize
			singB = scipy.ndimage.gaussian_filter(sing, sigma=sigma_center)
			singB = singB / np.max(singB)
		
			inp = np.zeros((dur,W,W))
			flipX = np.random.randint(0,2)
			flipY = np.random.randint(0,2)
			trans = np.random.randint(0,2)

			out = distort(singB,flipX,flipY,trans) #flip-transpose the output

			#os.system("mkdir spiral%03d%02d%01d" % (runN,j,flipX+2*flipY+4*trans))

			for i in range(dur):
					t = i + j*dur
					inp[i] = distort(imagesS[t],flipX,flipY,trans) #flip-transpose the input
					#misc.imsave("./spiral%03d%02d%01d/img_%01d.png" % (runN,j,flipX+2*flipY+4*trans,i),inp[i])

			#create a human-readable image: a spiral (first frame) overlayed with the position of the centre
			img[:, :, 0] = (1-out)*inp[0]*255 + out*255
			img[:, :, 1] = inp[0]*255
			img[:, :, 2] = inp[0]*255

			misc.imsave("./answers/visible_%03d%02d%01d.png" % (runN,j,flipX+2*flipY+4*trans),img)	#human-readable image to check that everything works fine
			misc.imsave("./answers/center_%03d%02d%01d.png" % (runN,j,flipX+2*flipY+4*trans),out)	#centre
			misc.imsave("./answers/sing_%03d%02d%01d.png" % (runN,j,flipX+2*flipY+4*trans),sing)	#one-hot

			with open("./pkls/spiral%03d%02d%01d.pkl" % (runN,j,flipX+2*flipY+4*trans), 'wb') as f:
				pkl.dump({"in" : inp, "out" : out}, f)

			#misc.imsave("./answers/visible_%03d%02d.png" % (runN,j),img)

		else:
			listS.append(0)	#fail
	if sum(listS)!=len(listS):
		print("%d done, %dp success" % (runN,sum(listS)/len(listS)*100))

