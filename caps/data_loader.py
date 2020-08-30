import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split


class Dataset:
    def __init__(self, dataset, _batch_size):
        super(Dataset, self).__init__()
        if dataset == 'sar_data':
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
            data = datasets.ImageFolder('../sar_data/', transform = transform) # dataset oluşturulması
            np.random.seed(42)
            train_size = int(0.70 * len(data)) # istenilen oranda datasetin train/test şeklinde bölünmesi
            test_size = len(data) - train_size
            train_dataset, test_dataset = random_split(data, [train_size, test_size])

            self.train_loader = DataLoader(train_dataset, batch_size= _batch_size, shuffle= True)

            self.test_loader = DataLoader(test_dataset, batch_size= _batch_size, shuffle= False)
