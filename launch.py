import sys
from readData import load_data
import matplotlib.pyplot as plt
from train import train


# just to show an image to show it is working, can be deleted at some point
def show_image(data_, labels_, batch, idx):
    print("label = ", labels_[batch][idx].item())
    plt.imshow(data_[batch][idx].cpu().permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    path = 'paintings'
    net = 'default'
    lr = 0.0002
    optimizerD = 'Adam'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        net = sys.argv[2]
    if len(sys.argv) > 3:
        lr = float(sys.argv[3])
    if len(sys.argv) > 4:
        optimizerD = sys.argv[4]

    if net == 'VANGAN':
        num_epochs = 550
    else:
        num_epochs = 250
    if len(sys.argv) > 5:
        num_epochs = int(sys.argv[5])

    # only testing if the networks handle the input correctly
    if path == 'test':
        from test_nets import test_generator, test_discriminator
        image = test_generator(net)
        test_discriminator(image, net)
        plt.imshow(image.detach().squeeze().permute(1, 2, 0))
        plt.show()
        exit(-1)

    # load the data in batches
    batch_size = 64
    dataloader, num_images = load_data(path=path, batch_size=batch_size, num_workers=8)
    print('data loaded')

    train(dataloader, num_epochs=num_epochs, net=net, learning_rate=lr, optimizerD=optimizerD,
          run_settings=path+'_'+net+'_'+str(lr)+'_'+optimizerD+'_'+str(num_epochs))
