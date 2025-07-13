import torchvision.transforms as transforms

# diff (range[-1,1]) to raw data
def diff_to_raw(dset):
    transform = transforms.Compose(
    [
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
    ]
    )
    return transform

# raw data to diff (range[-1,1]) 
def raw_to_diff(dset):
    transform = transforms.Compose(
    [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )
    return transform


def clf_to_raw(dset): 
    if dset in ["CIFAR10"]:
        # mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]  # [0.4914, 0.4824, 0.4467]
        # std = [63.0 / 255, 62.1 / 255, 66.7 / 255]  # [0.2471, 0.2435, 0.2616]
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose(
            [
                transforms.Normalize((-1.0*mean[0]/std[0], -1.0*mean[1]/std[1], -1.0*mean[2]/std[2]), (1./std[0], 1./std[1], 1./std[2])),
            ]
        )
        return transform
    elif dset in ["ImageNet","ImageNet-5k","ImageNet-Mini"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose(
            [
                transforms.Normalize((-1.0*mean[0]/std[0], -1.0*mean[1]/std[1], -1.0*mean[2]/std[2]), (1./std[0], 1./std[1], 1./std[2])),
            ]
        )
        return transform
    else:
        transform = transforms.Compose(
            [
            ]
        )
        return transform

def raw_to_clf(dset):
    if dset in ["CIFAR10"]:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose(
            [
                transforms.Normalize(mean, std)
            ]
        )
        return transform
    elif dset in ["ImageNet","ImageNet-5k","ImageNet-Mini"]:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        transform = transforms.Compose(
            [
                transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
            ]
        )
        return transform
    else:
        transform = transforms.Compose(
            [
            ]
        )
        return transform