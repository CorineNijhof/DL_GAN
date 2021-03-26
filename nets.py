import torch.nn as nn

# Number of channels in the training images. For color images this is 3
NC = 3
# Size of z latent vector (i.e. size of generator input)
NZ = 100
# Size of feature maps in generator
NGF = 512
# Size of feature maps in discriminator
NDF = 512

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(NZ, NGF * 8, 4, 8, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # state size. (NGF*8) x ? x ?
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 4, 0, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # state size. (NGF*4) x ? x ?
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 4, 0, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # state size. (NGF*2) x ? x ?
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 4, 0, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # state size. (NGF) x ? x ?
            nn.ConvTranspose2d(NGF, NC, 4, 2, 0, bias=False),
            nn.Tanh()
            # state size. (NC) x 512 x 512
        )

    def forward(self, input):
        return self.layers(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # input is (NC) x 512 x 512
            nn.Conv2d(in_channels=NC, out_channels=NDF, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF) x ? x ?
            nn.Conv2d(NDF, NDF * 2, 4, 4, 0, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*2) x ? x ?
            nn.Conv2d(NDF * 2, NDF * 4, 4, 4, 0, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*4) x ? x ?
            nn.Conv2d(NDF * 4, NDF * 8, 4, 4, 0, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x ? x ?
            nn.Conv2d(NDF * 8, 1, 4, 8, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)
