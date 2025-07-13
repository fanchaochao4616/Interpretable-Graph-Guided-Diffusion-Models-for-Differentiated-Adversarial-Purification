import os
import shutil
import argparse

import torch
import torchvision.transforms as transforms

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']

def diff2clf(x, is_imagenet=False): 
    # [-1, 1] to [0, 1]
    return torch.clamp((x / 2) + 0.5,0,1) 

def clf2diff(x):
    # [0, 1] to [-1, 1]
    return torch.clamp((x - 0.5) * 2,-1,1)

def normalize(x):
    # Normalization for ImageNet
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(x)

from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import sys
path_root = '/home/fcc/data/ensemble'
print("data_path_root:", path_root)

def import_data(dataset, train, shuffle, bsize, distortion_name="images",severity=None):
    '''
    dataset: datasets (MNIST, CIFAR10, CIFAR100, SVHN, CELEBA)
    train: True if training set, False if test set
    shuffle: Whether to shuffle or not
    bsize: minibatch size
    '''
    # Set transform
    dataset_list = ["MNIST", "CIFAR10", "ImageNet","ImageNet-Mini","ImageNet-5k","ImageNet-Mini"]
    if dataset not in dataset_list:
        sys.exit("Non-handled dataset")
    if dataset=="MNIST":
        path = os.path.join(path_root, "datasets", "MNIST")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = MNIST(path, train=train, download=True, transform=transform)
    elif dataset=="CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR10")
        dataset = CIFAR10(path, train=train, download=True, transform=transform)
    elif dataset=="ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "ImageNet")
        dataset = ImageNet(path, split='val', transform=transform)
    elif dataset=="ImageNet-Mini":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "ImageNet-Mini")
        valdir = path +'/' + distortion_name
        dataset = ImageFolder(valdir, transform)
        print(len(dataset))
    elif dataset=="ImageNet-5k":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        dataset = ImageFolder('/home/fcc/data/zhang_group/datasets/ImageNet_5k_label', transform)
    dataloader = DataLoader(
        dataset, batch_size=bsize, shuffle=shuffle, num_workers=4)
    return dataloader
