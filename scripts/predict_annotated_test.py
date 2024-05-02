import torch
import numpy as np
import torch.utils
import torch.utils.data
import nrrd
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import cKDTree
from torchvision.transforms import Resize
from mis.models import UNet2D
from mis.datasets import ASOCADataset
from mis.settings import DEVICE, ASOCA_PATH

# Copied from https://github.com/ramtingh/ASOCA_MICCAI2020_Evaluation/blob/master/evaluation.py
def hausdorff_95(submission,groundtruth, spacing):
    # There are more efficient algorithms for hausdorff distance than brute force, however, brute force is sufficient for datasets of this size.
    submission_points = spacing*np.array(np.where(submission), dtype=np.uint16).T
    submission_kdtree = cKDTree(submission_points)
    
    groundtruth_points = spacing*np.array(np.where(groundtruth), dtype=np.uint16).T
    groundtruth_kdtree = cKDTree(groundtruth_points)
    
    distances1,_ = submission_kdtree.query(groundtruth_points)
    distances2,_ = groundtruth_kdtree.query(submission_points)
    return max(np.quantile(distances1,0.95), np.quantile(distances2,0.95))

def dice(y_true, y_pred):
    return 2 * (y_true*y_pred).sum() / (y_true.sum()+y_pred.sum())

if __name__ == "__main__":

    size = 256
    data = ASOCADataset(
        size=size,
        split="test",
        merge_test_validation=False,
        two_dim=True,
        to_torch=True,
        norm=True,
        thresh=True,
        split_strat="patientwise"
    )

    healthy_idx = data.tih
    diseased_idx = data.tid

    model_dir = Path.cwd() / "unet2d_training_results_dice_asoca_tNone" / "model"
    model = UNet2D(1, 1).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))

    normal_dir = ASOCA_PATH / "Normal"
    diseased_dir = ASOCA_PATH / "Diseased" 

    out_dir = Path.cwd() / "annotest_preds"
    out_dir.mkdir(exist_ok=True)

    avg_dice = 0
    avg_haus = 0

    for idx in healthy_idx:
        img, _ = nrrd.read(normal_dir / "CTCA" / f"Normal_{idx}.nrrd")
        anno, meta = nrrd.read(normal_dir / "Annotations" / f"Normal_{idx}.nrrd")
        preds = np.zeros_like(img)

        fig, ax = plt.subplots(ncols=3)
        ims = []
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")

        print(f"Predicting healthy patient {idx}...")
        for slice_idx in tqdm(range(img.shape[-1])):
            ctca = img[:, :, slice_idx][None, :, :]
            ctca = ctca - ctca.min()
            ctca = ctca / np.abs(ctca).max()
            ctca = Resize((size, size))(torch.Tensor(ctca)).to(DEVICE)
            preds_nt = model(ctca[None,:,:,:])[-1]
            preds_nu = torch.where(preds_nt>=0.5, 1.0, 0.0)
            preds_u = F.interpolate(preds_nu, scale_factor=2, mode="nearest")
            preds[:,:,slice_idx] = preds_u.detach().cpu().numpy()
            
            im0 = ax[0].imshow(img[:, :, slice_idx].squeeze(), animated=True)
            ax[0].set_title("Image")
            im1 = ax[1].imshow(anno[:,:,slice_idx].squeeze(), animated=True)
            ax[1].set_title("Annotation")
            im2 = ax[2].imshow(preds[:,:,slice_idx].squeeze(), animated=True)
            ax[2].set_title(f"Prediction")
            ims.append([im0, im1, im2])

        print("Computing scores...")
        dice_score = dice(anno, preds)
        haus_score = hausdorff_95(preds, anno, np.diag(meta["space directions"]))
        print("Computing scores...")
        print(f"DICE={dice_score}")
        print(f"HAUS={haus_score}")

        fig.suptitle(f"Healthy Patient {idx} Prediction\nDice={dice_score}\nHausdorff={haus_score}")
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
        
        ani.save(out_dir / f"Healthy_{idx}.gif")
        plt.close()

        avg_dice += dice_score
        avg_haus += haus_score

    for idx in diseased_idx:
        img, _ = nrrd.read(diseased_dir / "CTCA" / f"Diseased_{idx}.nrrd")
        anno, meta = nrrd.read(diseased_dir / "Annotations" / f"Diseased_{idx}.nrrd")
        preds = np.zeros_like(img)

        fig, ax = plt.subplots(ncols=3)
        ims = []
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")

        print(f"Predicting diseased patient {idx}...")
        for slice_idx in tqdm(range(img.shape[-1])):
            ctca = img[:, :, slice_idx][None, :, :]
            ctca = ctca - ctca.min()
            ctca = ctca / np.abs(ctca).max()
            ctca = Resize((size, size))(torch.Tensor(ctca)).to(DEVICE)
            preds_nt = model(ctca[None,:,:,:])[-1]
            preds_nu = torch.where(preds_nt>=0.5, 1.0, 0.0)
            preds_u = F.interpolate(preds_nu, scale_factor=2, mode="nearest")
            preds[:,:,slice_idx] = preds_u.detach().cpu().numpy()
            
            im0 = ax[0].imshow(img[:, :, slice_idx].squeeze(), animated=True)
            ax[0].set_title("Image")
            im1 = ax[1].imshow(anno[:,:,slice_idx].squeeze(), animated=True)
            ax[1].set_title("Annotation")
            im2 = ax[2].imshow(preds[:,:,slice_idx].squeeze(), animated=True)
            ax[2].set_title(f"Prediction")
            ims.append([im0, im1, im2])

        print("Computing scores...")
        dice_score = dice(anno, preds)
        haus_score = hausdorff_95(preds, anno, np.diag(meta["space directions"]))
        print("Computing scores...")
        print(f"DICE={dice_score}")
        print(f"HAUS={haus_score}")

        fig.suptitle(f"Diseased Patient {idx} Prediction\nDice={dice_score}\nHausdorff={haus_score}")
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
        
        ani.save(out_dir / f"Diseased_{idx}.gif")
        plt.close()

        avg_dice += dice_score
        avg_haus += haus_score

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"dice_score": avg_dice, "hausdorff_95": avg_haus}, f)
