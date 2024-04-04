import torch
from pathlib import Path

BRATS_PATH = Path("/Users/eiriksteen/Personal/school/datasyn/TDT4265-Project-Track-2/data/brats2020")

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)