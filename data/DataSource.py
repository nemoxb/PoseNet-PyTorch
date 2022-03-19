from ctypes import resize
import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
from torch import from_numpy

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing
        # Resize Preprocess
        self.resize_tranforms = T.Resize((256,455))

        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image
        total = 0.0
        for img_path in self.images_path:
            img = Image.open(img_path)
            img = self.resize_tranforms(img)
            img = np.asarray(img,dtype=np.float32)
            total += img

        mean_image = total/len(self.images_path)
        np.save(self.mean_image_path,mean_image)

        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data = self.resize_tranforms(data)
        data = np.asarray(data,dtype=np.float32)
        data -= self.mean_image

        if self.train:
          transforms = T.Compose([ 
                                  T.ToTensor(),
                                  T.RandomCrop(self.crop_size),
                                  T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
          ])
        else:
          transforms = T.Compose([
                                  T.ToTensor(),
                                  T.CenterCrop(self.crop_size),
                                  T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
          ])
        data = transforms(data)
        return data, img_pose


    def __len__(self):
        return len(self.images_path)

# data_dir = '/Users/nemo/Documents/SFU/733/Assignments/A2/Assignment2-Py-Version/CMPT733-Lab2-Workspace/'+'data/datasets/KingsCollege/'
# trainset = DataSource(data_dir, train=True)
# train_loader = data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
# import matplotlib.pyplot as plt
# a, b= trainset.__getitem__(4)
# a = a.squeeze(0).cpu().numpy()
# plt.imshow(a,cmap='gray')
# plt.show()