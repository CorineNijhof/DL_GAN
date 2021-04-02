import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from nets import Generator, Discriminator, NZ
from run_nets import weights_init


# Number of GPUs available. Use 0 for CPU mode.
NGPU = 1


def train(dataloader, num_epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the nets
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netG = nn.DataParallel(generator, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Print the model
    # print(generator)
    # print(discriminator)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    learning_rate = 0.0002
    beta1 = 0.5

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 20 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving its output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i ==len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list.append(fake)

            iters += 1
            
    print("finished")
    for i in img_list:
        print("x")
    # print(img_list)
    
    print(img_list[-1][-1].size())
    plt.imshow(np.transpose(img_list[-1][-1],(1,2,0)))
    print(img_list[-1][-1].size())
    # plt.plot(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('fake_image_2.png')