import torch
import nrrd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from pathlib import Path
from ..settings import ASOCA_PATH

np.random.seed(0)

class ASOCADataset(Dataset):

    def __init__(
            self,
            size: int = 256,
            split: str = "train",
            merge_test_validation: bool = False,
            two_dim: bool = True,
            to_torch: bool = True,
            norm: bool = True,
            thresh: bool = True,
            data_dir: Path = ASOCA_PATH,
            split_strat: str = "random"
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

        self.num_normal = len(list((self.n_path / "CTCA").glob("Normal*")))
        self.num_diseased = len(list((self.d_path / "CTCA").glob("Diseased*")))
        self.merge_test_validation = merge_test_validation

        if split not in ["train", "validation", "test"]:
            raise ValueError("Split must be train or validation")
        
        if split == "test" and merge_test_validation:
            raise ValueError("No testset defined when merging test and validation")
        
        self.split = split

        strats =  ["random", "patientwise"]
        if split_strat not in ["random", "patientwise"]:
            raise ValueError(f"Split strat must be in {strats}")
        
        self.split_strat = split_strat

        if self.split_strat == "patientwise":
            self.trih, self.trid, self.vih, self.vid, self.tih, self.tid = self.get_patient_idx()

        self.num_slices, self.healthy_indices, self.diseased_indices = self.get_num_slices()
        self.slice_cumsums = np.asarray(self.num_slices).cumsum()

    def __len__(self):
        return sum(self.num_slices)

    def __getitem__(self, index):

        patient_idx = np.argwhere(self.slice_cumsums>index)[0,0]
        slice_idx = index % self.num_slices[patient_idx]

        if patient_idx < len(self.healthy_indices):
            ctca_path = self.n_path / "CTCA" /f"Normal_{self.healthy_indices[patient_idx]}.nrrd"
            anno_path = self.n_path / "Annotations" / f"Normal_{self.healthy_indices[patient_idx]}.nrrd"
        else:
            patient_idx -= len(self.healthy_indices)
            ctca_path = self.d_path / "CTCA" / f"Diseased_{self.diseased_indices[patient_idx]}.nrrd"
            anno_path = self.d_path / "Annotations" / f"Diseased_{self.diseased_indices[patient_idx]}.nrrd"

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
    
    def get_patient_idx(self):

        if self.merge_test_validation:   
            val_idx_healthy = (np.random.choice(self.num_normal, int(0.2 * self.num_normal), replace=False)+1).tolist()
            val_idx_diseased = (np.random.choice(self.num_diseased, int(0.2 * self.num_diseased), replace=False)+1).tolist()
            test_idx_healthy, test_idx_diseased = [], []
        else:
            idx_healthy = (np.random.choice(self.num_normal, int(0.4 * self.num_normal), replace=False)+1).tolist()
            idx_diseased = (np.random.choice(self.num_diseased, int(0.4 * self.num_diseased), replace=False)+1).tolist()
            mid = len(idx_healthy) // 2
            val_idx_healthy = idx_healthy[:mid]
            val_idx_diseased = idx_diseased[:mid]
            test_idx_healthy = idx_healthy[mid:]
            test_idx_diseased = idx_diseased[mid:]

        train_idx_healthy = [idx for idx in range(1, self.num_normal+1) if idx not in val_idx_healthy+test_idx_healthy]
        train_idx_diseased = [idx for idx in range(1, self.num_diseased+1) if idx not in val_idx_diseased+test_idx_diseased]

        return train_idx_healthy, train_idx_diseased, val_idx_healthy, val_idx_diseased, test_idx_healthy, test_idx_diseased


    def get_num_slices(self):

        num_slices = []
        healthy_indices = []
        diseased_indices = []

        if self.split_strat == "random":
            normal_idx = range(1,self.num_normal+1)
            diseased_idx = range(1,self.num_diseased+1)
        else:
            if self.split == "train":
                normal_idx = self.trih
                diseased_idx = self.trid
            elif self.split == "validation":
                normal_idx = self.vih
                diseased_idx = self.vid
            else:
                normal_idx = self.tih
                diseased_idx = self.tid

        for i in normal_idx:
            data, _ = nrrd.read(self.n_path / "CTCA" / f"Normal_{i}.nrrd")
            num_slices.append(data.shape[-1])
            healthy_indices.append(i)

        for i in diseased_idx:
            data, _ = nrrd.read(self.d_path / "CTCA" / f"Diseased_{i}.nrrd")
            num_slices.append(data.shape[-1])
            diseased_indices.append(i)

        return num_slices, healthy_indices, diseased_indices