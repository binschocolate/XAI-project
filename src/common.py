"""공통 유틸: 시드, 전처리, 데이터셋, 모델, 실행 이름."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models

# y: 0=landbird, 1=waterbird / place: 0=land, 1=water
GROUPS = {
    (0, 0): "landbird_land",    # aligned
    (0, 1): "landbird_water",   # conflicting
    (1, 0): "waterbird_land",   # conflicting
    (1, 1): "waterbird_water",  # aligned
}
ALIGNED = [(0, 0), (1, 1)]
CONFLICTING = [(0, 1), (1, 0)]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(hflip: bool = False) -> T.Compose:
    """기본: augmentation 없음. Resize 256 -> CenterCrop 224 -> ImageNet 정규화."""
    ops = [T.Resize((256, 256)), T.CenterCrop(224)]
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class WaterbirdsDataset(Dataset):
    """metadata 형식 DataFrame(img_filename, y, place)을 받아 이미지를 읽는다."""

    def __init__(self, df: pd.DataFrame, data_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.data_dir / r.img_filename).convert("RGB")
        return self.transform(img), int(r.y), int(r.place), i


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
    elif arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=w)
    else:
        raise ValueError(f"unknown arch: {arch}")
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def rho_tag(rho: float) -> str:
    """0.95 -> '095', 0.5 -> '050'"""
    return f"{round(rho * 100):03d}"


def run_name(arch: str, rho: float, seed: int) -> str:
    return f"{arch}_rho{rho_tag(rho)}_seed{seed}"


def group_counts(df: pd.DataFrame) -> dict:
    return {GROUPS[k]: int(((df.y == k[0]) & (df.place == k[1])).sum()) for k in GROUPS}
