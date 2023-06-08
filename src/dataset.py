import torch
from torch.utils.data import Dataset
from PIL import Image

class FlowersDataset(Dataset):
    def __init__(self, files_A, files_B, transform_A=None, transform_B=None, mode='train'):
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.files_A = files_A
        self.files_B = files_B
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Файлы не переданы"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform_A(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform_B(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()

        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))