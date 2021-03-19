from readData import load_data
import matplotlib.pyplot as plt


# just to show an image to show it is working, can be deleted at some point
def show_image(data, labels, batch, idx):
    print("label = ", labels[batch][idx].item())
    plt.imshow(data[batch][idx].cpu().permute(1, 2, 0))
    plt.show()

if __name__ == '__main__':
    data, labels = load_data('data') # all data, divided into batches
    # show_image(data, labels, 0, 0)