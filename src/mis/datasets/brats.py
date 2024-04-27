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
            to_torch=True,
            compression_params=None # [slice_num, offset]
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
        self.compression_params = compression_params

        if compression_params is not None:
            self.slice_num, self.offset = compression_params

        self.normalize = normalize
        self.to_torch = to_torch

    def __len__(self) -> int:
        return len(list(self.data_dir.glob("BraTS20*")))

    def __getitem__(self, index):

        patient_dir = self.data_dir / \
            list(self.data_dir.glob("BraTS20*"))[index]

        image_path = next(patient_dir.glob("*flair*"))
        seg_path = next(patient_dir.glob("*seg*"))
        t1_path = next(patient_dir.glob("*t1.*"))
        t1ce_path = next(patient_dir.glob("*t1ce*"))
        t2_path = next(patient_dir.glob("*t2*"))

        paths = [image_path, seg_path, t1_path, t1ce_path, t2_path]

        images = [nib.load(p).get_fdata().astype(np.float32) for p in paths]

        if self.compression_params is not None:
            images = [i[:,:,self.slice_num-self.offset:self.slice_num+self.offset] for i in images]
        
        if self.normalize:
            mask = images[1]
            images = [i / (np.abs(i).max() if np.abs(i).max() != 0 else 1) for i in images]
            images[1] = mask

        if self.to_torch:
            images = [i.transpose(2,0,1) for i in images]    
            images = [Resize((self.size, self.size))(torch.from_numpy(i)) for i in images]
            images = [i[None,:,:,:] for i in images]

        image, seg, t1, t1ce, t2 = images

        return {
            "image": image,
            "seg": seg.long(),
            "t1": t1,
            "t1ce": t1ce,
            "t2": t2
        }
