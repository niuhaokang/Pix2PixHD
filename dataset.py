import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

class ImageDataSet(data.Dataset):
    def __init__(self, data_root, img_size):
        super().__init__()
        self.data_root = data_root
        self.img_size = img_size

        self.imgs_A = [os.path.join(self.data_root, 'A', img)
                                   for img in os.listdir(os.path.join(self.data_root, 'A'))]
        self.imgs_B = [os.path.join(self.data_root, 'B', img)
                                   for img in os.listdir(os.path.join(self.data_root, 'B'))]

        assert len(self.imgs_A) == len(self.imgs_B)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.5), (0.5))
        ])

    def __getitem__(self, index):
        img_A = Image.open(self.imgs_A[index % len(self.imgs_A)])
        img_B = Image.open(self.imgs_B[index % len(self.imgs_B)])

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {
            'A' : img_A,
            'B' : img_B,
        }

    def __len__(self):
        return len(self.imgs_A)

class TestImageDataSet(data.Dataset):
    def __init__(self, data_root, img_size):
        super().__init__()
        self.imgs = [os.path.join(data_root, img) for img in os.listdir(data_root)]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        assert len(self.imgs) > 0

    def __getitem__(self, index):
        img_dir = self.imgs[index % len(self.imgs)]
        img = Image.open(img_dir)
        img = self.transform(img)
        return {
            'img': img
        }

    def __len__(self):
        return len(self.imgs)







