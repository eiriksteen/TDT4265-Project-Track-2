import torch
from pathlib import Path

BRATS_PATH = Path.cwd() / "data" / "brats2020"
ASOCA_PATH = Path.cwd() / "data" / "ASOCA"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"RUNNING ON DEVICE: {DEVICE}")