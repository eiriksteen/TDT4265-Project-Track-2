import torch
import nrrd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from pathlib import Path
from ..settings import ASOCA_PATH

class ASOCADataset(Dataset):

    def __init__(
            self,
            size: int = 256,
            two_dim: bool = True,
            to_torch: bool = True,
            norm: bool = True,
            thresh: bool = True,
            data_dir: Path = ASOCA_PATH,
        ):
        super().__init__()

        self.size = size
        self.two_dim = two_dim
        self.to_torch = to_torch
        self.norm = norm
        self.data_dir = data_dir
        self.n_path = data_dir / "Normal"
        self.d_path = data_dir / "Diseased"
        self.thresh = thresh
        self.num_slices, self.num_normal, self.num_diseased = self.get_num_slices()
        self.slice_cumsums = np.asarray(self.num_slices).cumsum()

    def __len__(self):
        return sum(self.num_slices)

    def __getitem__(self, index):
        patient_idx = np.argwhere(self.slice_cumsums>index)[0,0]
        slice_idx = index % self.num_slices[patient_idx]

        if patient_idx < self.num_normal:
            ctca_path = self.n_path / "CTCA" /f"Normal_{patient_idx+1}.nrrd"
            anno_path = self.n_path / "Annotations" / f"Normal_{patient_idx+1}.nrrd"
        else:
            ctca_path = self.d_path / "CTCA" / f"Diseased_{patient_idx%self.num_normal+1}.nrrd"
            anno_path = self.d_path / "Annotations" / f"Diseased_{patient_idx%self.num_normal+1}.nrrd"

        ctca, _ = nrrd.read(ctca_path)
        ctca = ctca[:,:,slice_idx][None, :, :]
        anno, _ = nrrd.read(anno_path)
        anno = anno[:,:,slice_idx][None, :, :]

        if self.norm:
            ctca = ctca - ctca.min()
            ctca = ctca / np.abs(ctca).max()

        if self.to_torch:
            ctca = Resize((self.size, self.size))(torch.Tensor(ctca))
            anno = Resize((self.size, self.size))(torch.Tensor(anno))
            if self.thresh:
                anno = torch.where(anno > 0.0, 1.0, 0.0)

        return {
            "image": ctca,
            "mask": anno
        }

    def get_num_slices(self):

        num_slices = []
        num_normal = len(list((self.n_path / "CTCA").glob("Normal*")))
        num_diseased = len(list((self.d_path / "CTCA").glob("Diseased*")))
        
        for i in range(num_normal):
            data, _ = nrrd.read(self.n_path / "CTCA" / f"Normal_{i+1}.nrrd")
            num_slices.append(data.shape[-1])

        for i in range(num_diseased):
            data, _ = nrrd.read(self.d_path / "CTCA" / f"Diseased_{i+1}.nrrd")
            num_slices.append(data.shape[-1])

        return num_slices, num_normal, num_diseased