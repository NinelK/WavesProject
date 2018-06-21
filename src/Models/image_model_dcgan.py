import torch
import torch.nn as nn 

def dcgan(inp=2,
          ndf=32,
          num_ups=4, need_sigmoid=True, need_bias=True, pad='zero', upsample_mode='nearest', need_convT = True):
    
    layers= [nn.ConvTranspose2d(inp, ndf, kernel_size=3, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(ndf),
             nn.LeakyReLU(True)]
    
    for i in range(num_ups-3):
        if need_convT:
            layers += [ nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ndf),
                        nn.LeakyReLU(True)]
        else:
            layers += [ nn.Upsample(scale_factor=2, mode=upsample_mode),
                        nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ndf),
                        nn.LeakyReLU(True)]
            
    if need_convT:
        layers += [nn.ConvTranspose2d(ndf, 1, 4, 2, 1, bias=False),]
    else:
        layers += [nn.Upsample(scale_factor=2, mode='bilinear'),
                   nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1, bias=False)]
    
    
    if need_sigmoid:
        layers += [nn.Sigmoid(), nn.Upsample(size=(86,86),mode='bilinear')]

    model =nn.Sequential(*layers)
    return model