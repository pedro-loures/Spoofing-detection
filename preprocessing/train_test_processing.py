import torch
import torchvision
from torchvision import datasets, transforms

def train_test_generator(train_dir= "data/train", test_dir= "data/test"):
    #transformations
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),                                
                                        torchvision.transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])

    #datasets
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    #dataloader
    train_loader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)

    return train_loader, test_loader

