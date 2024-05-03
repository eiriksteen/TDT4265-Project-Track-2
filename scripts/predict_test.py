import torch
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
import nrrd
import torch.nn.functional as F
from torchvision.transforms import Resize
from mis.models import UNet2D
from mis.settings import DEVICE, ASOCA_PATH

if __name__ == "__main__":
    """
    Tests the model on the test set of the ASOCA dataset to get a submission for the ASOCA Grand Challenge.
 
    Args:
        None: Alter the model_dir manually
 
    Returns:
        None: Saves the predictions to a directory
    """

    model_dir = Path.cwd() / "unet2d_training_results_dice_asoca_tNone" / "model"
    model = UNet2D(1, 1).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))

    normal_dir = ASOCA_PATH / "Normal" / "Testset_Normal"
    diseased_dir = ASOCA_PATH / "Diseased" / "Testset_Disease"

    out_dir = Path.cwd() / "test_preds"
    out_dir.mkdir(exist_ok=True)

    size = 256
    for i in range(10):
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

        np.savez_compressed(out_dir/f"{i}")

    for j in range(10, 20):
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

        np.savez_compressed(out_dir/f"{j}")
