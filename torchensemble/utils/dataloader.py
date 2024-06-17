from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FixedDataLoader(object):
    def __init__(self, dataloader):
        # Check input
        if not isinstance(dataloader, DataLoader):
            msg = (
                "The input used to instantiate FixedDataLoader should be a"
                " DataLoader from `torch.utils.data`."
            )
            raise ValueError(msg)

        self.elem_list = []
        for _, elem in enumerate(dataloader):
            self.elem_list.append(elem)

    def __getitem__(self, index):
        return self.elem_list[index]

    def __len__(self):
        return len(self.elem_list)


def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


def load_cifar10(data_dir, batch_size=128, split=False):
    train_transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    if split:
        train_idx = [0, 1, 2, 3, 4, 5]
        ood_idx = [6, 7, 8, 9]

    else:
        train_idx = list(range(10))

    trainset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transformer)
    train_subset = Subset(trainset, get_classes(trainset, train_idx))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )

    testset = datasets.CIFAR10(data_dir, train=False, transform=test_transformer)
    test_subset = Subset(testset, get_classes(testset, train_idx))
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=True
    )

    # oodset = datasets.CIFAR10(data_dir, train=False, transform=test_transformer)
    if split:
        ood_subset = Subset(testset, get_classes(testset, ood_idx))
        ood_loader = DataLoader(
            ood_subset,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        ood_loader = None

    return train_loader, test_loader, ood_loader
