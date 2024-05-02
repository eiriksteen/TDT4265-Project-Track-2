import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import seaborn as sns
import torch.utils
import torch.utils.data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import nrrd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize
from mis.models import UNet2D, UNet2DNonLocal
from mis.datasets import ASOCADataset, BratsDataset
from mis.settings import DEVICE, ASOCA_PATH
from mis.loss import dice_loss, gdlv_loss, focal_loss

if __name__ == "__main__":

    # model_dir = Path.cwd() / "unet2d_training_results_dice_asoca" / "model"
    model_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts") / "unet2d_training_results_dice_asoca_tNone" / "model"
    model = UNet2D(1, 1).to(DEVICE)

    normal_dir = ASOCA_PATH / "Normal" / "Testset_Normal"
    diseased_dir = ASOCA_PATH / "Diseased" / "Testset_Disease"

    # out_dir = Path.cwd() / "test_preds"
    out_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts") / "test_preds"
    out_dir.mkdir(exist_ok=True)

    size = 256
    for i in range(1):
        img, _ = nrrd.read(normal_dir / f"{i}.nrrd")
        preds = np.zeros_like(img)
        print(f"Predicting patient {i+1}...")
        for slice_idx in tqdm(range(img.shape[-1])):
            ctca = img[:, :, slice_idx][None, :, :]
            ctca = ctca - ctca.min()
            ctca = ctca / np.abs(ctca).max()
            ctca = Resize((size, size))(torch.Tensor(ctca)).to(DEVICE)
            preds_nt = model(ctca[None,:,:,:])[-1]
            preds_nu = torch.where(preds_nt>=0.5, 1.0, 0.0)
            preds_u = F.interpolate(preds_nu, scale_factor=2, mode="nearest")
            preds[:,:,slice_idx] = preds_u.detach().cpu().numpy()

        filename_nrrd = str(out_dir/f"{i}.nrrd")
        nrrd.write(filename_nrrd, preds)
        np.savez_compressed(out_dir/f"{i}")

    for j in range(10, 11):
        img, _ = nrrd.read(diseased_dir / f"{j}.nrrd")
        preds = np.zeros_like(img)
        print(f"Predicting patient {j+1}...")
        for slice_idx in tqdm(range(img.shape[-1])):
            ctca = img[:, :, slice_idx][None, :, :]
            ctca = ctca - ctca.min()
            ctca = ctca / np.abs(ctca).max()
            ctca = Resize((size, size))(torch.Tensor(ctca)).to(DEVICE)
            preds_nt = model(ctca[None,:,:,:])[-1]
            preds_nu = torch.where(preds_nt>=0.5, 1.0, 0.0)
            preds_u = F.interpolate(preds_nu, scale_factor=2, mode="nearest")
            preds[:,:,slice_idx] = preds_u.detach().cpu().numpy()

        filename_nrrd = str(out_dir/f"{j}.nrrd")
        nrrd.write(filename_nrrd, preds)
        np.savez_compressed(out_dir/f"{j}")
