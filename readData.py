import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# load the image data and labels into lists of tensors
def load_data(path, batch_size=25, num_workers=8):
    torch.manual_seed(1)
    # resize image, 255 chosen at random atm
    transformer = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform=transformer)
    n_images = len(dataset)
    # num_worker for parallel, pin_memory should improve speed later on
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

    data = []
    labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for image_batch, label_batch in dataloader:
        # transfer drawings to gpu
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        data.append(image_batch)
        labels.append(label_batch)
    
    return dataloader, data, labels, n_images
