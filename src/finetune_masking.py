"""마스킹 적응 추가 학습 실험 — 기존 checkpoint 에서 두 조건으로 추가 학습.

질문: 기존 순위 불안정성이 "모델이 마스킹 입력에 낯설어서" 생긴 것인가?
  control   : 원본 이미지만으로 추가 학습 (단순히 학습을 더 한 효과의 대조군)
  maskadapt : 원본 + 무작위 patch 를 baseline 으로 가린 이미지로 추가 학습
두 조건은 시작 checkpoint, 학습 이미지(B안 split 1,678장), epoch, 배치 순서, optimizer 가 모두 같다.

사전 고정 설정 (2026-11-23, 결과 보기 전에 결정)
  - 대상: ResNet-18 clean/shortcut × seed 0–2 (6개)
  - 추가 학습: 5 epoch, SGD lr 1e-4, momentum 0.9, wd 1e-4, batch 32, augmentation 없음 (원래 학습과 같은 전처리)
  - maskadapt: 배치의 각 이미지를 확률 0.5 로 마스킹 (원본과 마스킹 이미지를 섞음)
      · 가리는 patch 수: 14×14 grid 196개 중 이미지마다 10–30% 균등 (판단 정보가 충분히 남는 범위)
      · 위치: XAI 와 무관한 무작위 patch
      · 채우는 값: 평가와 같은 baseline 5종 (black/mean/blur/inpaint/trainset) 중 이미지마다 균등 무작위.
        trainset donor = 같은 배치의 다른 학습 이미지 (test 이미지는 쓰지 않음)
  - 배치 순서는 seed 로 고정해 두 조건이 같음. 마스킹 난수는 별도 generator 라 배치 순서에 영향 없음
출력: <out_root>/<cond>/<run>/{final.pt, config.json, train_log.csv}
      config.json 에 arch·rho·seed·run_name 이 있어 팀 src/eval_sri.py 로 바로 평가 가능

사용: python src/finetune_masking.py --data_dir $DATA_DIR --run resnet18_rho095_seed0 --cond maskadapt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from attribution import SIZE, load_model
from common import IMAGENET_MEAN, IMAGENET_STD, WaterbirdsDataset, get_transform, set_seed
from faithfulness import BASELINES, push_pull_inpaint, train_pixel_mean

REPO = Path(__file__).resolve().parent.parent


def random_keep(n, grid, fmin, fmax, gen, device):
    """이미지마다 fmin–fmax 비율의 무작위 patch 를 지우는 keep mask (n,1,H,W), 1=남김"""
    n_patch = grid * grid
    k = torch.randint(int(round(fmin * n_patch)), int(round(fmax * n_patch)) + 1, (n,), generator=gen)
    rank = torch.rand(n, n_patch, generator=gen).argsort(1).argsort(1)  # patch 별 무작위 순위
    keep = (rank >= k[:, None]).float().view(n, 1, grid, grid)
    return F.interpolate(keep.to(device), size=(SIZE, SIZE), mode="nearest"), k


def apply_baselines(xp, keep, kinds, pix_mean, donors, blur_k, blur_s):
    """xp: (n,3,H,W) 픽셀, kinds: (n,) baseline index -> 마스킹된 픽셀 이미지 (단계 6 과 같은 구현)"""
    out = xp.clone()
    for bi, b in enumerate(BASELINES):
        sel = kinds == bi
        if not sel.any():
            continue
        x, kp = xp[sel], keep[sel]
        if b == "inpaint":
            out[sel] = push_pull_inpaint(x, kp, pix_mean)  # 이미 원본 부분과 합성된 결과
            continue
        fill = {"black": lambda: torch.zeros_like(x), "mean": lambda: pix_mean.expand_as(x),
                "blur": lambda: TF.gaussian_blur(x, blur_k, blur_s), "trainset": lambda: donors[sel]}[b]()
        out[sel] = x * kp + (1 - kp) * fill
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run", required=True, help="예: resnet18_rho095_seed0")
    ap.add_argument("--cond", required=True, choices=["control", "maskadapt"])
    ap.add_argument("--ckpt_root", default=str(REPO / "models"))
    ap.add_argument("--out_root", default=str(REPO / "models_ft"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--p_mask", type=float, default=0.5)
    ap.add_argument("--frac_min", type=float, default=0.10)
    ap.add_argument("--frac_max", type=float, default=0.30)
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--blur_kernel", type=int, default=41)
    ap.add_argument("--blur_sigma", type=float, default=8.0)
    ap.add_argument("--mask_seed", type=int, default=2026)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    arch, rho_tag, seed_tag = args.run.split("_")
    seed = int(seed_tag.replace("seed", ""))
    out = Path(args.out_root) / args.cond / args.run
    if (out / "final.pt").exists() and not args.overwrite:
        print(f"[skip] {out}/final.pt exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(REPO / "splits" / f"train_{rho_tag}_{seed_tag}.csv")
    g = torch.Generator().manual_seed(seed)  # 배치 순서: 두 조건 동일
    loader = DataLoader(WaterbirdsDataset(train_df, args.data_dir, get_transform()), batch_size=args.batch_size,
                        shuffle=True, generator=g, num_workers=args.num_workers, pin_memory=True)
    mgen = torch.Generator().manual_seed(args.mask_seed + seed)  # 마스킹 난수 (maskadapt 에서만 사용)

    model = load_model(args.run, args.ckpt_root, device)
    model.requires_grad_(True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    pix_mean = train_pixel_mean(args.data_dir, REPO / "outputs" / "faithfulness" / "train_pixel_mean.json",
                                args.num_workers).to(device).view(3, 1, 1)

    cfg = dict(arch=arch, rho=0.95 if rho_tag == "rho095" else 0.5, seed=seed, run_name=args.run, condition=args.cond,
               init_checkpoint=str(Path(args.ckpt_root) / args.run / "final.pt"), n_train=len(train_df),
               **{k: v for k, v in vars(args).items() if k not in ("data_dir", "run", "cond", "overwrite")},
               baselines=BASELINES if args.cond == "maskadapt" else [], device=device)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(json.dumps(cfg, ensure_ascii=False), flush=True)

    log = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n, correct, nm, cm = time.time(), 0.0, 0, 0, 0, 0
        for x, y, _, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            masked = torch.zeros(len(y), dtype=torch.bool)
            if args.cond == "maskadapt":
                masked = torch.rand(len(y), generator=mgen) < args.p_mask
                if masked.any():
                    xp = x * std + mean
                    sel = masked.to(device)
                    keep, _ = random_keep(int(masked.sum()), args.grid, args.frac_min, args.frac_max, mgen, device)
                    kinds = torch.randint(0, len(BASELINES), (int(masked.sum()),), generator=mgen).to(device)
                    donors = xp.roll(1, 0)[sel]  # 같은 배치의 다른 학습 이미지
                    xp[sel] = apply_baselines(xp[sel], keep, kinds, pix_mean, donors, args.blur_kernel, args.blur_sigma)
                    x = (xp - mean) / std
            out_ = model(x)
            loss = crit(out_, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hit = (out_.argmax(1) == y).cpu()
            tot += loss.item() * len(y)
            n += len(y)
            correct += hit.sum().item()
            nm += masked.sum().item()
            cm += hit[masked].sum().item()
        row = dict(epoch=ep, loss=tot / n, acc=correct / n, n_masked=nm, acc_masked=cm / nm if nm else float("nan"),
                   sec=round(time.time() - t0, 1))
        log.append(row)
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    torch.save(model.state_dict(), out / "final.pt")
    print(f"[done] {out}/final.pt", flush=True)


if __name__ == "__main__":
    main()
