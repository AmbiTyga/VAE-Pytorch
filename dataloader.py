import torch.utils.data as data
import torch
import h5py
import numpy as np
import pdb

class GraphVAEDataLoader(data.Dataset):
    def __init__(self,
                 h5f,
                 Tensor,
                 transform=None,
                 return_mask=False):
        self.hf = h5f
        self.transform = transform
        self.Tensor = Tensor
        self.return_mask = return_mask

    def __getitem__(self, index):
        return_dict = dict()
        # NOTE: the data is already normalized between [0,1]
        img_tensor = torch.tensor(self.hf['data'][index]).type(self.Tensor)
        return_dict['image'] = img_tensor.unsqueeze(0)

        return_dict['tuple'] = self.hf['label'][index]
        if self.return_mask:
            return_dict['mask'] = torch.tensor(
                    self.hf['masks'][index]).type(self.Tensor)
        return return_dict

    def __len__(self):
        return len(self.hf['data'])





