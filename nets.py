'''
Based on PyTorch's tutorial on DCGANs: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

import torch.nn as nn

# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64


class Generator(nn.Module):
    def __init__(self, net):
        super(Generator, self).__init__()

        if net == 'default128':
            self.layers = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),

                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # for 128x128
                # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),   # for 64x64
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 32 x 32
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 64 x 64
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 128 x 128
            )
        elif net == 'default':
            self.layers = nn.Sequential(
                # input is Z, going into a convolution

                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),   # for 64x64
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        elif net == 'VANGAN':
            self.layers = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 8, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 14 x 14
                nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 26 x 26
                nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 50 x 50
                nn.ConvTranspose2d(ngf * 8, ngf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 52 x 52
                nn.ConvTranspose2d(ngf * 8, ngf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 54 x 54
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                # state size. (ngf*8) x 56 x 56
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                # state size. (ngf*8) x 58 x 58
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                # state size. (ngf) x 61 x 61
                nn.ConvTranspose2d(ngf, nc, 4, 1, 0, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, input):
        return self.layers(input)


class Discriminator(nn.Module):
    def __init__(self, net):
        super(Discriminator, self).__init__()

        if net == 'default':
            self.layers = nn.Sequential(
                # input is (nc) x 128 x 128

                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 64 x 64

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # for 128x128
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # for 128x128
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # for 128x128
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8

                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),  # for 128x128
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                ## state size. (ndf*8) x 4 x 4

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif net == 'default':
            self.layers = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # for 64x64
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # for 64x64
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # for 64x64
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # for 64x64
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                ## state size. (ndf*8) x 4 x 4

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif net == 'VANGAN':
            # · Convolution: selects weights for the next layer based on a filter of weights from the previous layer
            # · Dropout: zeros out weights randomly to prevent overfitting
            # · Batch Normalization: normalizes the weights as a form of regularization
            # · Leaky ReLU: allows for nonzero gradients to prevent overfitting
            # · Max Pooling: dimensionality reduction
            self.layers = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf * 4, 8, 4, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 15 x 15
                nn.Conv2d(ndf * 4, ndf * 8, 6, 3, 0, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.layers(input)
