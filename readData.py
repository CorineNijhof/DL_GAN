import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# load the image data and labels into lists of tensors
def load_data(path, batch_size, num_workers=8, size=64):
    torch.manual_seed(1)
    # resize image, 64,64 is the default
    if size == 128:
    	transformer = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    else: 
    	transformer = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform=transformer)
    n_images = len(dataset)
    # num_worker for parallel, pin_memory should improve speed later on
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    
    return dataloader, n_images
