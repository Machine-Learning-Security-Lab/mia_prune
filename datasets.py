import pandas as pd
import numpy as np
import sklearn
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import TensorDataset


def get_dataset(name, train=True):
    print(f"Build Dataset {name}")
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='/data/datasets/cifar10-data', train=train, download=True, transform=transform)
    elif name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='/data/datasets/cifar100-data', train=train, download=True,
                                                transform=transform)
    elif name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])
        dataset = torchvision.datasets.MNIST(root='/data/datasets/mnist-data', train=train, download=True,
                                             transform=transform)

    elif name == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = torchvision.datasets.SVHN(root='/data/datasets/svhn-data', split='train' if train else "test",
                                            download=True, transform=transform)

    elif name == "texas100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
        dataset = np.load("/data/datasets/texas/data_complete.npz")
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_data = torch.tensor(dataset['y'][:] - 1).long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None

    elif name == "location":
        # the dataset can be downloaded from https://github.com/jjy1994/MemGuard/tree/master/data/location
        dataset = np.load("/data/datasets/location/data_complete.npz")
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_data = torch.tensor(dataset['y'][:] - 1).long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None

    elif name == "purchase100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz
        tensor_path = "/data/datasets/purchase100/purchase100.pt"
        if os.path.exists(tensor_path):
            data = torch.load(tensor_path)
            x_data, y_data = data['x'], data['y']
        else:
            dataset = np.loadtxt("/data/datasets/purchase100/purchase100.txt", delimiter=',')
            x_data = torch.tensor(dataset[:, :-1]).float()
            y_data = torch.tensor(dataset[:, - 1]).long()
            torch.save({'x': x_data, 'y': y_data}, tensor_path)
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            dataset = None

    elif name == "chmnist":
        # the dataset can be downloaded from https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1
        data_folder = "/data/datasets/chmnist/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000"
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if train:
            dataset = torchvision.datasets.ImageFolder(data_folder, transform=transform)
        else:
            dataset = None
    else:
        raise ValueError

    return dataset

