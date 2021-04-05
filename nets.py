import torch.nn as nn

# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 50
# Size of feature maps in generator
ngf = 128
# Size of feature maps in discriminator
ndf = 128


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

        # · Transpose Convolution: “deconvolution”
        # · ReLU
        # · Batch Normalization
        # · Max Pooling
        # self.network = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 16),
        #     nn.LeakyReLU(True),
        #     # nn.MaxUnpool2d(2),
        #     # state size. (ngf*16) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.LeakyReLU(True),
        #     # state size. (ngf*8) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.LeakyReLU(True),
        #     # state size. (ngf*4) x 16 x 16
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.LeakyReLU(True),
        #     # state size. (ngf*2) x 32 x 32
        #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.LeakyReLU(True),
        #     # state size. (ngf) x 64 x 64
        #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 128 x 128
        # )

    def forward(self, input):
        x = self.network(input)
        # print('generator output size:', x.size())
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.network = nn.Sequential(
        #     # input is (NC) x 128 x 128
        #     nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (NDF) x 64 x 64
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (NDF*2) x 32 x 32
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (NDF*4) x 16 x 16
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (NDF*8) x 8 x 8
        #     nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (NDF*16) x 4 x 4
        #     nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

        # · Convolution: selects weights for the next layer based on a filter of weights from the previous layer
        # · Dropout: zeros out weights randomly to prevent overfitting
        # · Batch Normalization: normalizes the weights as a form of regularization
        # · Leaky ReLU: allows for nonzero gradients to prevent overfitting
        # · Max Pooling: dimensionality reduction
        self.network = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 64 x 64
            nn.MaxPool2d(kernel_size=2),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.MaxPool2d(kernel_size=2),

            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 4 x 4
            nn.MaxPool2d(kernel_size=2),

            # state size. (ndf*4) x 2 x 2
            nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # x = self.network1(input)
        # print(x.size())
        # x = self.network2(x)
        # print(x.size())
        # x = self.network3(x)
        # print(x.size())
        # return x
        return self.network(input)
