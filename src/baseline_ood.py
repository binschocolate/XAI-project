"""Masking baseline 별 OOD 정도 정량화 (proposal: "baseline별 OOD 정도를 정량화").

아이디어: **같은 patch 를 같은 비율로 지우고 채우는 값(baseline)만 바꾼다** → 지운 정보량은 같으므로
baseline 간 차이는 채운 값의 효과. 두 축을 따로 잰다.
  ① OOD 정도   : 마스킹 이미지의 특징이 학습 분포에서 얼마나 벗어났나
                  - Mahalanobis 거리 (Lee et al. 2018; 클래스별 평균 + 공통 공분산, Ledoit-Wolf shrinkage)
                  - KNN 거리 (Sun et al. 2022; L2 정규화 특징, k 번째 최근접 학습 이미지까지 거리)
                  - AUROC(원본 test 이미지 vs 마스킹 이미지): 0.5 = 구분 불가(ID), 1.0 = 완전히 구분(OOD)
  ② 정보 잔존  : 마스킹 이미지가 원본 정보를 얼마나 유지하나
                  - 원본 특징과의 cosine 유사도, 원본 예측과의 일치율, 원본 예측 class 확률
특징 = 마지막 avgpool 출력 (R18 512, R50 2048 차원).

모델
  - 우리 12개 모델: 각자의 B안 학습 split(splits/train_rho*_seed*.csv, 1,678장)이 학습 분포 → 모델이 느끼는 OOD (H3-2)
  - --runs imagenet_resnet50: Waterbirds 로 학습하지 않은 ImageNet ResNet-50, 학습 분포 = train+val pool 5,994장,
    클래스 없는 단일 Gaussian Mahalanobis → 모델과 무관한 "이미지 자체의 부자연스러움" 기준선

마스크: 이미지마다 고정 seed 의 무작위 patch 순서(14×14 grid, 196 patch)에서 앞 k 개 제거, 비율 25/50/75%.
       baseline 구현은 단계 6(src/faithfulness.py)과 동일 (black / mean / blur / inpaint / trainset, 같은 donor).

출력 (outputs/baseline_ood/<run>/)
  ood.csv      baseline × 비율 × scope(all/aligned/conflicting) 요약 (+ 비율 0 = 원본 기준값)
  scores.npz   이미지별 maha / knn / cos / pred / p_target (원본 포함)
  config.json
사용: python src/baseline_ood.py --data_dir $DATA_DIR --runs resnet18_rho095_seed0
     python src/baseline_ood.py --data_dir $DATA_DIR --runs imagenet_resnet50
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from attribution import SIZE, load_model
from common import IMAGENET_MEAN, IMAGENET_STD, WaterbirdsDataset, get_transform, set_seed
from faithfulness import BASELINES, load_trainset_fills, push_pull_inpaint, train_pixel_mean

REPO = Path(__file__).resolve().parent.parent
REF = "imagenet_resnet50"


# ----------------------------------------------------------------------------- features
class FeatModel:
    """avgpool 특징 + logits"""

    def __init__(self, model):
        self.model, self.store = model, {}
        model.avgpool.register_forward_hook(lambda m, i, o: self.store.__setitem__("f", o.flatten(1)))

    @torch.no_grad()
    def __call__(self, x):
        logits = self.model(x)
        return self.store["f"].float(), logits.float()


def extract(fm, loader, device):
    fs, ls = [], []
    for x, *_ in loader:
        f, l = fm(x.to(device))
        fs.append(f.cpu())
        ls.append(l.cpu())
    return torch.cat(fs).numpy(), torch.cat(ls).numpy()


# ----------------------------------------------------------------------------- OOD scores
class OODScorer:
    def __init__(self, ftr, ytr, k):
        self.k = k
        classes = [None] if ytr is None else sorted(set(ytr))
        self.mu = np.stack([ftr.mean(0) if c is None else ftr[ytr == c].mean(0) for c in classes])
        centered = ftr - (self.mu[0] if ytr is None else self.mu[np.searchsorted(classes, ytr)])
        self.prec = LedoitWolf().fit(centered).precision_
        self.tr_n = ftr / np.linalg.norm(ftr, axis=1, keepdims=True)

    def maha(self, f):
        return np.min([np.einsum("ij,jk,ik->i", f - m, self.prec, f - m) for m in self.mu], 0)

    def knn(self, f):
        fn = f / np.linalg.norm(f, axis=1, keepdims=True)
        d = 2 - 2 * fn @ self.tr_n.T  # 정규화 특징의 squared L2
        return np.sort(d, 1)[:, self.k - 1]


def auroc(orig, masked):
    """마스킹 이미지가 원본보다 OOD 점수가 높을 확률 (0.5 = 구분 불가)"""
    return roc_auc_score(np.r_[np.zeros(len(orig)), np.ones(len(masked))], np.r_[orig, masked])


# ----------------------------------------------------------------------------- masking
def random_keep_masks(attr_idx, ks, grid, seed, device):
    """이미지별 고정 seed 무작위 patch 순서 -> (len(ks),1,H,W) keep mask (1=남김)"""
    n_patch = grid * grid
    perm = np.random.default_rng(seed + int(attr_idx)).permutation(n_patch)
    rank = np.empty(n_patch, int)
    rank[perm] = np.arange(n_patch)
    keep = torch.as_tensor((rank[None, :] >= np.array(ks)[:, None]).astype(np.float32))
    keep = keep.view(len(ks), 1, grid, grid).to(device)
    return F.interpolate(keep, size=(SIZE, SIZE), mode="nearest")


def fill(base, xp, keep, pix_mean, fill_train, args):
    if base == "black":
        return xp * keep
    if base == "mean":
        return xp * keep + (1 - keep) * pix_mean
    if base == "blur":
        return xp * keep + (1 - keep) * TF.gaussian_blur(xp[None], args.blur_kernel, args.blur_sigma)
    if base == "trainset":
        return xp * keep + (1 - keep) * fill_train
    if base == "inpaint":
        return push_pull_inpaint(xp[None].expand(keep.shape[0], -1, -1, -1), keep, pix_mean)
    raise ValueError(base)


# ----------------------------------------------------------------------------- main
def run_one(run, args, df, xs_pix, fills, pix_mean, device, mean, std):
    out = Path(args.out_root) / run
    if (out / "ood.csv").exists() and not args.overwrite:
        print(f"[{run}] exists, skip (--overwrite)", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = run == REF
    if ref:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
        meta = pd.read_csv(Path(args.data_dir) / "metadata.csv")
        tr = meta[meta.split.isin([0, 1])][["img_filename", "y", "place", "split"]]
    else:
        model = load_model(run, args.ckpt_root, device)
        arch, rho, seed = run.split("_")
        tr = pd.read_csv(REPO / "splits" / f"train_{rho}_{seed}.csv")
    fm = FeatModel(model)

    # 학습 분포 특징
    tl = DataLoader(WaterbirdsDataset(tr, args.data_dir, get_transform()), batch_size=128, num_workers=args.num_workers)
    ftr, _ = extract(fm, tl, device)
    scorer = OODScorer(ftr, None if ref else tr.y.values, args.k)
    print(f"[{run}] train features {ftr.shape} ({time.time() - t0:.0f}s)", flush=True)

    # 원본 test(xai subset) 특징
    fs, ls = [], []
    for i in range(0, len(xs_pix), 128):
        f, l = fm((xs_pix[i:i + 128].to(device) - mean) / std)
        fs.append(f.cpu().numpy())
        ls.append(l.cpu().numpy())
    f0, l0 = np.concatenate(fs), np.concatenate(ls)
    pred0 = l0.argmax(1)
    p0 = torch.softmax(torch.as_tensor(l0), 1).numpy()
    maha0, knn0 = scorer.maha(f0), scorer.knn(f0)

    n_patch = args.grid * args.grid
    ks = [int(round(fr * n_patch)) for fr in args.fractions]
    nB, nF, nI = len(BASELINES), len(ks), len(df)
    maha = np.zeros((nB, nF, nI), np.float32)
    knn, cos, agree, ptar = (np.zeros_like(maha) for _ in range(4))
    for i in range(nI):
        xp = xs_pix[i].to(device)
        keep = random_keep_masks(df.attr_idx.values[i], ks, args.grid, args.mask_seed, device)
        batch = torch.cat([fill(b, xp, keep, pix_mean, fills[i].to(device) if fills is not None else None, args)
                           for b in BASELINES])  # (nB*nF,3,H,W)
        f, l = fm((batch - mean) / std)
        f, l = f.cpu().numpy(), l.cpu().numpy()
        maha[:, :, i] = scorer.maha(f).reshape(nB, nF)
        knn[:, :, i] = scorer.knn(f).reshape(nB, nF)
        cos[:, :, i] = (f @ f0[i] / (np.linalg.norm(f, axis=1) * np.linalg.norm(f0[i]) + 1e-12)).reshape(nB, nF)
        agree[:, :, i] = (l.argmax(1) == pred0[i]).reshape(nB, nF)
        ptar[:, :, i] = torch.softmax(torch.as_tensor(l), 1).numpy()[:, pred0[i]].reshape(nB, nF)
        if (i + 1) % 200 == 0 or i + 1 == nI:
            print(f"[{run}] {i + 1}/{nI} ({time.time() - t0:.0f}s)", flush=True)

    g = df.y.values * 2 + df.place.values
    scopes = {"all": np.ones(nI, bool), "aligned": np.isin(g, [0, 3]), "conflicting": np.isin(g, [1, 2])}
    rows = []
    for sc, m in scopes.items():
        rows.append(dict(run=run, baseline="original", fraction=0.0, scope=sc, n=int(m.sum()), auroc_maha=0.5,
                         auroc_knn=0.5, maha=maha0[m].mean(), knn=knn0[m].mean(), cos_to_orig=1.0,
                         pred_agree=1.0, p_target=p0[np.arange(nI), pred0][m].mean()))
        for bi, b in enumerate(BASELINES):
            for fi, fr in enumerate(args.fractions):
                rows.append(dict(run=run, baseline=b, fraction=fr, scope=sc, n=int(m.sum()),
                                 auroc_maha=auroc(maha0[m], maha[bi, fi][m]), auroc_knn=auroc(knn0[m], knn[bi, fi][m]),
                                 maha=maha[bi, fi][m].mean(), knn=knn[bi, fi][m].mean(), cos_to_orig=cos[bi, fi][m].mean(),
                                 pred_agree=agree[bi, fi][m].mean(), p_target=ptar[bi, fi][m].mean()))
    res = pd.DataFrame(rows)
    res.to_csv(out / "ood.csv", index=False)
    np.savez_compressed(out / "scores.npz", maha=maha, knn=knn, cos=cos, agree=agree, p_target=ptar,
                        maha_orig=maha0, knn_orig=knn0, pred_orig=pred0, baselines=np.array(BASELINES),
                        fractions=np.array(args.fractions), img_filename=df.img_filename.values, group=g)
    cfg = dict(run=run, reference_model=ref, n_train=len(tr), feat_dim=int(ftr.shape[1]), knn_k=args.k,
               maha="class-conditional means + shared Ledoit-Wolf covariance, min over classes" if not ref
               else "single Gaussian (no classes), Ledoit-Wolf", fractions=args.fractions, grid=args.grid,
               mask="per-image random patch order, seed = mask_seed + attr_idx", mask_seed=args.mask_seed,
               blur={"kernel": args.blur_kernel, "sigma": args.blur_sigma}, mean_pixel=pix_mean.flatten().tolist(),
               trainset_seed=args.trainset_seed, n_images=nI, seconds=round(time.time() - t0, 1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"[{run}] done ({cfg['seconds']}s)\n" +
          res[(res.scope == "all") & (res.fraction == 0.5)][["baseline", "auroc_maha", "auroc_knn", "cos_to_orig",
                                                             "pred_agree", "p_target"]].round(3).to_string(index=False),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help=f"모델 run 이름들 또는 {REF}")
    ap.add_argument("--ckpt_root", default=str(REPO / "models"))
    ap.add_argument("--split", default=str(REPO / "splits" / "test_xai800.csv"))
    ap.add_argument("--out_root", default=str(REPO / "outputs" / "baseline_ood"))
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--k", type=int, default=10, help="KNN 의 k (학습 이미지 1,678장 기준)")
    ap.add_argument("--mask_seed", type=int, default=1000)
    ap.add_argument("--blur_kernel", type=int, default=41)
    ap.add_argument("--blur_sigma", type=float, default=8.0)
    ap.add_argument("--trainset_seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="그룹별 limit/4 장만 (동작 확인용, 결과 해석 금지)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.split)
    df["attr_idx"] = np.arange(len(df))
    n_full = len(df)
    if args.limit:
        df = df.groupby(["y", "place"], group_keys=False).head(max(1, args.limit // 4)).reset_index(drop=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    dl = DataLoader(WaterbirdsDataset(df, args.data_dir, get_transform()), batch_size=64, num_workers=args.num_workers)
    xs_pix = torch.cat([x for x, *_ in dl]) * std.cpu() + mean.cpu()
    pix_mean = train_pixel_mean(args.data_dir, REPO / "outputs" / "faithfulness" / "train_pixel_mean.json",
                                args.num_workers).to(device)
    all_fills, _ = load_trainset_fills(args.data_dir, n_full, args.trainset_seed, args.num_workers)  # 단계 6 과 같은 donor
    fills = all_fills[torch.as_tensor(df.attr_idx.values)]
    print(f"device={device} images={len(df)} runs={args.runs}", flush=True)
    for run in args.runs:
        run_one(run, args, df, xs_pix, fills, pix_mean, device, mean, std)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
