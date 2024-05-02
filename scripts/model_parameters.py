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
from mis.models import UNet2D
from mis.settings import DEVICE, ASOCA_PATH
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_parameters_unet(model_name):
    
    # model_dir = Path.cwd() / "unet2d_training_results_dice_asoca_tNone" / "model"     # For Mac
    scripts_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts")
    model_dir = scripts_dir / f"{model_name}" / "model"
    
    model = UNet2D(1, 1).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))

    num_model_parameters = count_parameters(model)
    print(f"Number of model parameters for {model_name}: {num_model_parameters}")

def get_model_parameters_segformer(model_name):
    
    # model_dir = Path.cwd() / "unet2d_training_results_dice_asoca_tNone" / "model"     # For Mac
    scripts_dir = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/scripts")
    model_dir = scripts_dir / f"{model_name}" / "model"
    
    config = SegformerConfig()
    config.id2label = {0: "background", 1: "artery"}
    config.label2id = {"background": 0, "artery": 1}
    config.semantic_loss_ignore_index = 0
    config.num_channels = 1

    model = SegformerForSemanticSegmentation(config).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location="cpu"))

    num_model_parameters = count_parameters(model)
    print(f"Number of model parameters for {model_name}: {num_model_parameters}")
    
if __name__ == "__main__":
    
    model_names = [
        "unet2d_results_dice_concat_t",
        "unet2d_results_dice_sum_t",
        "segformer_training_results_dice_asoca_base"
    ]
    
    for model_name in model_names:
        if "unet" in model_name:
            get_model_parameters_unet(model_name)
        if "segformer" in model_name:
            get_model_parameters_segformer(model_name)
        else:
            ValueError("Model name not recognized")