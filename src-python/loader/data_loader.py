from torch.utils.data import Dataset
import os
import time 

import numpy as np 
import torch 
import torch.optim as optim


class SlicesDataset(Dataset):
  """
  This class represents an indexable Torch dataset
  which could be consumed by the PyTorch DataLoader class
  """
  def __init__(self, data):
    self.data = data
    self.slices = []

    for i, d in enumerate(data):
      for j in range(d["image"].shape[0]):
        self.slices.append((i, j))
  
  def __getitem__(self, idx):
    """
    This method is called by Pytorch DataLoader class to return a sample with id idx
    """

    slc = self.slices[idx]
    sample = dict()
    sample["id"] = idx

    # We could implement caching strategy here if dataset is too large to fit in memory

    sample["image"] = torch.tensor(self.data[slc[0]]["image"][slc[1]]).unsqueeze(0)
    sample["seg"] = torch.tensor(self.data[slc[0]]["seg"][slc[1]]).unsqueeze(0)

    return sample
  
  def __len__(self):
    return len(self.slices)