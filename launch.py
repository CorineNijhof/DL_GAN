from readData import load_data
import matplotlib.pyplot as plt
from run_nets import run_generator, run_discriminator


# just to show an image to show it is working, can be deleted at some point
def show_image(data_, labels_, batch, idx):
    print("label = ", labels_[batch][idx].item())
    plt.imshow(data_[batch][idx].cpu().permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    data, labels = load_data('data', num_workers=6)  # all data, divided into batches
    # show_image(data, labels, 0, 0)
    print('data loaded')

    image = run_generator()

    run_discriminator(image)
