import torch
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
import nrrd
import torch.nn.functional as F
from torchvision.transforms import Resize
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from mis.models import UNet2D, UNet2DNonLocal
from mis.settings import DEVICE, ASOCA_PATH
import time

def get_model_unet2d(model_name):
    
    # model_dir = Path.cwd() / "unet2d_training_results_dice_asoca_tNone" / "model"     # For Mac
    scripts_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts")
    model_dir = scripts_dir / f"{model_name}" / "model"
    
    if "nonlocal" in model_name:
        if "concat" in model_name:
            model = UNet2DNonLocal(1, 1, skip_conn="concat").to(DEVICE)
        else:
            model = UNet2DNonLocal(1, 1, skip_conn="sum").to(DEVICE)
    else:
        model = UNet2D(1, 1).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))
    return model

def get_model_segformer(model_name):
    
    # model_dir = Path.cwd() / "unet2d_training_results_dice_asoca_tNone" / "model"     # For Mac
    scripts_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts")
    model_dir = scripts_dir / f"{model_name}" / "model"
    
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))
    return model
    
if __name__ == "__main__":
    
    model_names = [
        "segformer_results_dice_patientwise_low_lr",
        "segformer_results_dice_t_patientwise",
    ]
    
    for model_name in model_names:
        if "unet2d" in model_name:
            model = get_model_unet2d(model_name)
        if "segformer" in model_name:
            model = get_model_segformer(model_name)
        else:
            ValueError("Model name not recognized")
            
        num_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model parameters for {model_name}: {num_model_parameters}")