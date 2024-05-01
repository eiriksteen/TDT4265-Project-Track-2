import torch
import torch.nn.functional as F

def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor):

    return 1 - (2 * (y_true*y_pred).sum() + 1) / ((y_true**2).sum()+(y_pred**2).sum() + 1)

def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor):

    return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()

def gdlv_loss(y_true: torch.Tensor, y_pred: torch.Tensor):

    r0, p0 = (1 - y_true).flatten(start_dim=1), (1 - y_pred).flatten(start_dim=1)
    r1, p1 = y_true.flatten(start_dim=1), y_pred.flatten(start_dim=1)

    w0 = torch.zeros(r0.shape[0]).to(y_pred.device)
    w1 = torch.zeros(r1.shape[0]).to(y_pred.device)
    idx0 = r0.sum(dim=1)>0
    idx1 = r1.sum(dim=1)>0

    w0[idx0] = (1 / r0.sum(dim=1)**2)[idx0] 
    w1[idx1] = (1 / r1.sum(dim=1)**2)[idx1] 

    numer = w0*(r0*p0).sum(dim=1) + w1*(r1*p1).sum(dim=1)
    denom = w0*(r0+p0).sum(dim=1) + w1*(r1+p1).sum(dim=1)

    gdl = (1 - 2 * numer / denom).mean()    

    return gdl


def focal_loss(y_true: torch.Tensor, y_pred: torch.Tensor, gamma: float = 2.0):

    logpt = F.binary_cross_entropy(y_pred, y_true)
    pt = torch.e ** (-logpt)

    return torch.mean((1 - pt)**gamma * logpt)

# def soft_dice(y_true: torch.Tensor, y_pred: torch.Tensor):

#     dice = 1 - 2 * (y_true * y_pred) / (y_true*y_true + y_pred*y_pred)