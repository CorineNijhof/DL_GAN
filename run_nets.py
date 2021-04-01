from nets import Generator, Discriminator, NC, NZ, NGF, NDF
from torch.optim import Adam
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_image(generator, fixed_noise):
    generated_img = generator(fixed_noise).cpu().detach()
    print(generated_img.size())

    plt.imshow(generated_img.squeeze().permute(1, 2, 0))
    plt.show()
    return generated_img


def run_generator():
    # torch.cuda.is_available = lambda : False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    generator = Generator().to(device)

    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    # netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    fixed_noise = torch.randn(1, NZ, 1, 1, device=device)
    # # fixed_noise = torch.randn(64, 64, 1, 1, device=device)
    # optimizer_generator = Adam(generator.parameters())
    # criterion = nn.BCELoss()

    generator.apply(weights_init)

    # Print the model
    print(generator)

    return show_image(generator, fixed_noise)



def run_discriminator(input):
    # torch.cuda.is_available = lambda : False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    discriminator = Discriminator().to(device)

    # optimizer_generator = Adam(generator.parameters())
    # criterion = nn.BCELoss()

    discriminator.apply(weights_init)

    # Print the model
    print(discriminator)

    vector = discriminator(input)
    print('output of discriminator is:', vector.item())




