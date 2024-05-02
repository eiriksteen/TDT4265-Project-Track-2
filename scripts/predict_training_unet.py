import torch
import torch.nn as nn
import numpy as np
import argparse
import shutil
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
    model_name = "unet2d_training_results_dice_asoca_tNone"
    model_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts") / f"{model_name}" / "model"
    model = UNet2D(1, 1).to(DEVICE)

    # Get training data
    normal_img_dir = ASOCA_PATH / "Normal" / "CTCA"
    diseased_img_dir = ASOCA_PATH / "Diseased" / "CTCA"
    normal_label_dir = ASOCA_PATH / "Normal" / "Annotations"
    diseased_label_dir = ASOCA_PATH / "Diseased" / "Annotations"

    # out_dir = Path.cwd() / "test_preds"
    out_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts") / "training_preds"
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir / f"{model_name}"
    out_dir.mkdir(exist_ok=True)
    out_dir_normal = out_dir / "Normal"
    out_dir_diseased = out_dir / "Diseased"
    out_dir_normal.mkdir(exist_ok=True)
    out_dir_diseased.mkdir(exist_ok=True)

    size = 256
    for i in range(1):
        img, _ = nrrd.read(normal_img_dir / f"Normal_{i+1}.nrrd")
        preds = np.zeros_like(img)
        print(f"Predicting patient {i+1}...")
        for slice_idx in tqdm(range(img.shape[-1])[:2]):
            ctca = img[:, :, slice_idx][None, :, :]
            ctca = ctca - ctca.min()
            ctca = ctca / np.abs(ctca).max()
            ctca = Resize((size, size))(torch.Tensor(ctca)).to(DEVICE)
            preds_nt = model(ctca[None,:,:,:])[-1]
            preds_nu = torch.where(preds_nt>=0.5, 1.0, 0.0)
            preds_u = F.interpolate(preds_nu, scale_factor=2, mode="nearest")
            preds[:,:,slice_idx] = preds_u.detach().cpu().numpy()

        filename_preds_nrrd = str(out_dir_normal/f"{i+1}_preds.nrrd")
        filename_preds_seg_nrrd = str(out_dir_normal/f"{i+1}_preds.seg.nrrd")
        nrrd.write(filename_preds_nrrd, preds)
        nrrd.write(filename_preds_seg_nrrd, preds)
        
        shutil.copyfile(src=normal_img_dir / f"Normal_{i+1}.nrrd", dst=out_dir_normal / f"{i+1}_img.nrrd")
        shutil.copyfile(src=normal_label_dir / f"Normal_{i+1}.nrrd", dst=out_dir_normal / f"{i+1}_label.nrrd")
        shutil.copyfile(src=normal_label_dir / f"Normal_{i+1}.nrrd", dst=out_dir_normal / f"{i+1}_label.seg.nrrd")
        

    for j in range(1):
        img, _ = nrrd.read(diseased_img_dir / f"Diseased_{j+1}.nrrd")
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

        filename_preds_nrrd = str(out_dir_diseased/f"{j+1}_preds.nrrd")
        filename_preds_seg_nrrd = str(out_dir_diseased/f"{j+1}_preds.seg.nrrd")
        nrrd.write(filename_preds_nrrd, preds)
        nrrd.write(filename_preds_seg_nrrd, preds)

        shutil.copyfile(src=diseased_img_dir / f"Normal_{j+1}.nrrd", dst=out_dir_diseased / f"{j+1}_img.nrrd")
        shutil.copyfile(src=diseased_label_dir / f"Normal_{j+1}.nrrd", dst=out_dir_diseased / f"{j+1}_label.nrrd")
        shutil.copyfile(src=diseased_label_dir / f"Normal_{j+1}.nrrd", dst=out_dir_diseased / f"{j+1}_label.seg.nrrd")