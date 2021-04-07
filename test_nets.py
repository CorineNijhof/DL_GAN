from nets import Generator, Discriminator, nc, nz, ngf, ndf
from torch.optim import Adam
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from train import weights_init


def show_image(generator, fixed_noise):
    generated_img = generator(fixed_noise).cpu().detach()
    # print(generated_img.size())

    # plt.imshow(generated_img.squeeze().permute(1, 2, 0))
    # plt.show()
    return generated_img


def test_generator(net):
    # torch.cuda.is_available = lambda : False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    generator = Generator(net).to(device)

    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    # netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    # # fixed_noise = torch.randn(64, 64, 1, 1, device=device)
    # optimizer_generator = Adam(generator.parameters())
    # criterion = nn.BCELoss()

    generator.apply(weights_init)

    # Print the model
    print(generator)
    return generator(fixed_noise)
    # return show_image(generator, fixed_noise)


def test_discriminator(input, net):
    # torch.cuda.is_available = lambda : False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    discriminator = Discriminator(net).to(device)

    # optimizer_generator = Adam(generator.parameters())
    # criterion = nn.BCELoss()

    discriminator.apply(weights_init)

    # Print the model
    print(discriminator)

    vector = discriminator(input)
    print('output of discriminator is:', vector.item())
