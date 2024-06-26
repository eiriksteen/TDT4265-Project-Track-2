import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import Resize
from ..settings import BRATS_PATH


class BratsDataset(Dataset):

    def __init__(
            self,
            split: str,
            data_dir: Path = BRATS_PATH,
            size: int = 128,
            normalize=True,
            to_torch=True
    ) -> None:
        super().__init__()

        if split not in ["train", "validation"]:
            raise ValueError("Split must be train or validation")
        elif split == "train":
            data_dir = data_dir / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
        else:
            data_dir = data_dir / "BraTS2020_ValidationData" / \
                "MICCAI_BraTS2020_ValidationData"

        self.data_dir = data_dir
        self.size = size
        self.normalize = normalize
        self.to_torch = to_torch

    def __len__(self) -> int:
        return 155 * len(list(self.data_dir.glob("BraTS20*")))

    def __getitem__(self, index):

        dir_idx = index // 155
        slice_num = index % 155

        patient_dir = self.data_dir / \
            list(self.data_dir.glob("BraTS20*"))[dir_idx]
        
        image_path = next(patient_dir.glob("*flair*"))
        
        try:
            seg_path = next(patient_dir.glob("*seg*"))
        except StopIteration:
            seg_path = next(patient_dir.glob("*Seg*"))

        t1_path = next(patient_dir.glob("*t1.*"))
        t1ce_path = next(patient_dir.glob("*t1ce*"))
        t2_path = next(patient_dir.glob("*t2*"))

        paths = [image_path, seg_path, t1_path, t1ce_path, t2_path]
        images = [nib.load(p).get_fdata()[:,:,slice_num][:,:,None].astype(np.float32) for p in paths]
        
        if self.normalize:
            mask = images[1]
            images = [i / (np.abs(i).max() if np.abs(i).max() != 0 else 1) for i in images]
            images[1] = mask

        if self.to_torch:
            images = [i.transpose(2,0,1) for i in images]    
            images = [Resize((self.size, self.size))(torch.from_numpy(i)) for i in images]

        image, seg, t1, t1ce, t2 = images

        seg = torch.where(seg>0,1,0).float()

        return {
            "image": image,
            "mask": seg,
            "t1": t1,
            "t1ce": t1ce,
            "t2": t2
        }
