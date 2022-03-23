import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import glob

class CorruptMNIST(Dataset):
    def __init__(self, npz_files, transform=None):
        self.transform = transform
        self.labels = None
        self.images = None
        for file in npz_files:
            with np.load(file, allow_pickle=True) as data:      
                # labels_length = len(data['labels'])
                labels_tmp = np.array(data['labels'])
                if self.labels is None:
                    self.labels = labels_tmp.copy()
                else:
                    self.labels = np.append(self.labels, labels_tmp)
                
                # images_shape = data['images'].shape
                images_tmp = np.array(data['images'])
                if self.images is None:
                    self.images = images_tmp.copy()
                else:
                    self.images = np.append(self.images, images_tmp, axis=0)                

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

    train_files = glob.glob('data/corruptmnist/train*.npz')
    train = CorruptMNIST(train_files, transform)

    test_files = glob.glob('data/corruptmnist/test.npz')
    test = CorruptMNIST(test_files, transform)

    return train, test
