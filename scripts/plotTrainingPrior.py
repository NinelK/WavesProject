import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib import pylab as plt
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import animation
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, DATA_DIR
from src.Dataset import get_image
import torch

def plot_training_loss(experiment_name='Test', filename='training_loss.dat'):
    filename = os.path.join(LOG_DIR, experiment_name, 'training_loss.dat')
    loss = []
    with open(filename) as fin:
        for line in fin:
            sline = line.split()
            loss.append(float(sline[1]))
    plt.figure(figsize=(12,8))
    plt.plot(np.array(loss))
    plt.show()
    

def plot_images(experiment_name='Test', init_image_name=''):
    import operator
    images = {}
    exp_dir = os.path.join(LOG_DIR, experiment_name)
    for filename in os.listdir(exp_dir):
        if filename.find('image')!=-1:
            epoch = int(filename.split('_')[1].split('.')[0])
            image = torch.load(os.path.join(exp_dir,filename))
            images[epoch] = image
    print images.keys()
    epoch_list, img_list = zip(*sorted(images.items(), key=operator.itemgetter(0)))
    img_list = list(img_list)
    print epoch_list

    plt.figure(figsize=(3*len(images.keys()),3))
    big_image = torch.cat(img_list, dim=3)
    plt.imshow(big_image[0,0,:,:].numpy())
    plt.tight_layout()
    plt.savefig(experiment_name+'_all.png')

    max_epoch = max(images.keys())
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(images[epoch][0,0,:,:].numpy())
    
    plt.subplot(1,2,2)
    init_image = get_image(os.path.join(DATA_DIR,init_image_name))[0,0,:,:].cpu().numpy()
    plt.imshow(init_image)
    plt.savefig(experiment_name+'.png')
    # plt.show()

if __name__=='__main__':
    experiment_name='ResNetSpiral5'
    plot_training_loss(experiment_name=experiment_name)
    plot_images(experiment_name=experiment_name, init_image_name='SpiralsInPetri/Spiral5_circ.tif')

