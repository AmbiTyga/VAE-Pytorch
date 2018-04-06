import torch.utils.data as data
import h5py 
import numpy as np
import glob
import cv2
import pdb

class GetDataset(data.Dataset):
    def __init__(self, path, transform = None):
        self.filenames = glob.glob(path)
        self.transform = transform 

    def __getitem__(self, index):
        img_path = self.filenames[index]
        # img = cv2.imread(img_path)/255.
        img = cv2.imread(img_path)
        # img = np.transpose(img, (2, 0, 1))
        if self.transform is not None:
            img = self.transform(img)
        return img.view(1, -1)
    def __len__(self):
        return len(self.filenames)

