import torch
from pathlib import Path

BRATS_PATH = Path.cwd() / "data" / "brats2020"
# ASOCA_PATH = Path.cwd() / "data" / "ASOCA"
ASOCA_PATH = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/coronary-artery-segmentation/data/asoca")

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"RUNNING ON DEVICE: {DEVICE}")