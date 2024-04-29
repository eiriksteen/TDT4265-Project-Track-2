import torch
from pathlib import Path

BRATS_PATH = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/TDT4265-Project-Track-2/data/brats2020")
# ASOCA_PATH = Path.cwd() / "ASOCA"
ASOCA_PATH = Path("C:/Users/henri/Desktop/NTNU/4.Året/Vår/TDT 4265 - Computer Vision/TDT4265-Computer-Vision/TDT4265-Project-Track-2/data/asoca")     # Fix this

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"RUNNING ON DEVICE: {DEVICE}")