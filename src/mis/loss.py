import torch

def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    n = len(y_true.flatten())
    return (1/n) * (torch.sum(1 - (2*y_true*y_pred + 1)/(y_true+y_pred + 1)))