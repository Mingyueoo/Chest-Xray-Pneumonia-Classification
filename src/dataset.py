from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = (128, 128)


def build_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def get_dataloader(data_dir, batch_size=32, shuffle=False):
    dataset = ImageFolder(root=data_dir, transform=build_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset.classes
