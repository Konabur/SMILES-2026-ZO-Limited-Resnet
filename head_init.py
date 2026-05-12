"""
head_init.py — Final layer initialization via linear probing.

Computes a closed-form ridge-regression solution on frozen ImageNet
features extracted from a CIFAR100 train subset, then copies the result
into the new 100-class head. No gradients of the backbone are computed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from augmentation import get_transforms

_NUM_CLASSES = 100
_PROBE_SAMPLES = 4096
_PROBE_BATCH = 64
_RIDGE_LAMBDA = 1.0
_DATA_DIR = "./data"


def _extract_features(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    dataset = datasets.CIFAR100(
        root=_DATA_DIR,
        train=True,
        download=True,
        transform=get_transforms(train=False),
    )
    indices = list(range(min(_PROBE_SAMPLES, len(dataset))))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=_PROBE_BATCH,
        shuffle=False,
        num_workers=0,
    )

    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the head via ridge-regression linear probing.

    Extracts frozen ImageNet features on a deterministic CIFAR100 train
    subset, then solves W = (XᵀX + λI)⁻¹ XᵀY in closed form, where Y is
    the one-hot label matrix. The bias absorbs the means of X and Y so the
    solution is centred.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = _extract_features(device)

    Y = F.one_hot(y, _NUM_CLASSES).float()
    x_mean = X.mean(dim=0, keepdim=True)
    y_mean = Y.mean(dim=0, keepdim=True)
    Xc = X - x_mean
    Yc = Y - y_mean

    feat_dim = X.shape[1]
    A = Xc.T @ Xc + _RIDGE_LAMBDA * torch.eye(feat_dim)
    B = Xc.T @ Yc
    W = torch.linalg.solve(A, B)
    b = (y_mean - x_mean @ W).squeeze(0)

    with torch.no_grad():
        layer.weight.copy_(W.T.to(layer.weight.device, layer.weight.dtype))
        layer.bias.copy_(b.to(layer.bias.device, layer.bias.dtype))
