import torch
from torch.utils.data import DataLoader
from mis.datasets import ASOCADataset
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = ASOCADataset(
        size=256,
        two_dim=True,
        to_torch=True,
        norm=True
    )

    train_dl = DataLoader(data, batch_size=8, shuffle=True)

    for sample in data:
        mask = sample["mask"]
        zero_count = len(mask[mask==0].flatten())
        one_count = len(mask[mask==1].flatten())
        if one_count != 0:
            print("@"*20)
            print(f"ZC: {zero_count}")
            print(f"OC: {one_count}")
            if zero_count != 0  :
                print(f"R1: {zero_count/one_count}")
                print(f"R2: {one_count/zero_count}")
