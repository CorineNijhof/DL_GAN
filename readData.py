import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# load the image data and labels into lists of tensors
def load_data(path):
    torch.manual_seed(1)
    # resize image, 255 chosen at random atm
    transformer = transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform=transformer)
    # num_worker for parallel, pin_memory should improve speed later on
    dataloader = DataLoader(dataset, batch_size=250, num_workers=6, pin_memory=True, shuffle=True)

    data = []
    labels = []
    for image_batch, label_batch in dataloader:
        if torch.cuda.is_available():
            # transfer data to gpu
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()
        data.append(image_batch)
        labels.append(label_batch)
    
    return data, labels
