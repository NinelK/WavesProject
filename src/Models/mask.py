import numpy as np
from torch import from_numpy
from torch.autograd import Variable

def get_mask(W=86,numpy=False):
    '''
    Creates a Petri disk mask with diameter W (px)
    '''
    xx, yy = np.ogrid[:W,:W]
    sel = np.array((xx-(W-1)/2)**2 + (yy-(W-1)/2)**2 < (43)**2).astype("float32")
    if numpy:
    	return sel
    else:
	    mask = from_numpy(sel)
	    return Variable(mask.cuda())
