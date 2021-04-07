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
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        net = sys.argv[2]

    # only testing if the networks handle the input correctly
    if path == 'test':
        from test_nets import test_generator, test_discriminator
        image = test_generator(net)
        test_discriminator(image, net)
        exit(-1)

    # load the data in batches
    batch_size = 64
    dataloader, num_images = load_data(path=path, batch_size=batch_size, num_workers=6)
    print('data loaded')

    train(dataloader, num_epochs=150, net=net)
