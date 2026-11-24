"""H3 실험 (GPU, 쌍 단위: 같은 arch·seed 의 clean–shortcut). 설계: docs/H3_DESIGN.md

  실험 A (clean 모델만) : 배경을 baseline 5종으로 채움 / 같은 장소 자연 배경으로 교체 / 반대 장소로 교체 → Δp
  H3-2 단계 1           : 4×4 칸마다 자연 배경(물 4 + 땅 4)으로 교체 → 두 모델 각각 칸별 max|Δp|
  H3-2 단계 2           : τ ∈ {0.02, 0.05} 별 공통 안 보는 칸 C 를 baseline 5종으로 채움 / 같은 장소 자연 배경으로 교체
                          → 두 모델 Δp, 특징 cosine
  OOD                   : 모든 개입 이미지의 ImageNet ResNet-50 Mahalanobis 거리 (원본 대비 AUROC 는 분석 단계에서)
Δp = p(원본, 원래 예측 class) − p(개입 후, 같은 class)
출력: outputs/h3/pairs/<arch>_seed<k>/raw.npz, config.json
사용: python src/h3_experiment.py --data_dir $DATA_DIR --arch resnet18 --seed 0 [--limit 16]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from attribution import load_model
from baseline_ood import FeatModel, OODScorer, fill
from common import IMAGENET_MEAN, IMAGENET_STD, WaterbirdsDataset, get_transform, set_seed
from faithfulness import BASELINES, load_trainset_fills, train_pixel_mean

REPO = Path(__file__).resolve().parent.parent
TAUS = [0.02, 0.05]
GRID, CELL = 4, 56


class Net:
    """정규화 + 특징(avgpool) + softmax 확률"""

    def __init__(self, model, mean, std):
        self.fm, self.mean, self.std = FeatModel(model), mean, std

    @torch.no_grad()
    def __call__(self, xp, chunk=256):
        fs, ps = [], []
        for i in range(0, len(xp), chunk):
            f, l = self.fm((xp[i:i + chunk] - self.mean) / self.std)
            fs.append(f)
            ps.append(torch.softmax(l, 1))
        return torch.cat(fs), torch.cat(ps)


def cell_mask(c, device):
    """칸 c (0..15) 만 1 인 (1,1,224,224)"""
    m = torch.zeros(1, 1, 224, 224, device=device)
    r, k = divmod(c, GRID)
    m[..., r * CELL:(r + 1) * CELL, k * CELL:(k + 1) * CELL] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--arch", required=True, choices=["resnet18", "resnet50"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prep", default=str(REPO / "outputs" / "h3" / "prep"))
    ap.add_argument("--out_root", default=str(REPO / "outputs" / "h3" / "pairs"))
    ap.add_argument("--ckpt_root", default=str(REPO / "models"))
    ap.add_argument("--limit", type=int, default=None, help="그룹별 limit/4 장 (동작 확인용, 결과 해석 금지)")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    out = Path(args.out_root) / f"{args.arch}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    prep = Path(args.prep)
    # 파일명 배열이 object 형식으로 저장돼 있어 allow_pickle 필요 (우리가 만든 파일만 읽음)
    tm, bank, don = (np.load(prep / f, allow_pickle=True) for f in ("test_masks.npz", "bank.npz", "donors.npz"))

    df = pd.read_csv(REPO / "splits" / "test_xai800.csv")
    df["idx"] = np.arange(len(df))
    assert (tm["img_filename"] == df.img_filename.values).all()
    if args.limit:
        df = df.groupby(["y", "place"], group_keys=False).head(max(1, args.limit // 4)).reset_index(drop=True)
    idx = df.idx.values
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    dl = DataLoader(WaterbirdsDataset(df, args.data_dir, get_transform()), batch_size=64, num_workers=args.num_workers)
    xs = torch.cat([x for x, *_ in dl]) * std.cpu() + mean.cpu()  # 픽셀 [0,1]
    bird = torch.from_numpy(tm["bird"][idx]).float()[:, None]  # (n,1,224,224)
    bank_pix = torch.from_numpy(bank["pix"]).float() / 255  # (128,3,224,224)
    same, opp = don["same"][idx], don["opp"][idx]
    pix_mean = train_pixel_mean(args.data_dir, REPO / "outputs" / "faithfulness" / "train_pixel_mean.json",
                                args.num_workers).to(dev).view(3, 1, 1)
    fills, _ = load_trainset_fills(args.data_dir, len(tm["img_filename"]), 0, args.num_workers)  # 단계 6 과 같은 donor
    fills = fills[torch.as_tensor(idx)]
    fa = SimpleNamespace(blur_kernel=41, blur_sigma=8.0)

    runs = {"clean": f"{args.arch}_rho050_seed{args.seed}", "shortcut": f"{args.arch}_rho095_seed{args.seed}"}
    nets = {k: Net(load_model(r, args.ckpt_root, dev), mean, std) for k, r in runs.items()}
    ref = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(dev).eval()
    refnet = Net(ref, mean, std)

    # ImageNet 기준 OOD scorer (train+val pool 5,994장, 단일 Gaussian)
    meta = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    pool = meta[meta.split.isin([0, 1])][["img_filename", "y", "place", "split"]]
    pl = DataLoader(WaterbirdsDataset(pool, args.data_dir, get_transform()), batch_size=128, num_workers=args.num_workers)
    with torch.no_grad():
        fpool = torch.cat([refnet.fm(x.to(dev))[0].cpu() for x, *_ in pl]).numpy()
    scorer = OODScorer(fpool, None, k=10)
    print(f"[{args.arch} seed{args.seed}] ready: {len(df)} images, ref pool {fpool.shape} ({time.time() - t0:.0f}s)", flush=True)

    n, nB, nD = len(df), len(BASELINES), same.shape[1]
    R = {}  # 결과 배열
    # 원본
    f0, p0, tgt = {}, {}, {}
    for k, net in nets.items():
        f, p = net(xs.to(dev))
        tgt[k] = p.argmax(1)
        p0[k] = p.gather(1, tgt[k][:, None])[:, 0].cpu().numpy()
        f0[k] = f
    R["ref_maha_orig"] = scorer.maha(refnet(xs.to(dev))[0].cpu().numpy())
    for k in nets:
        R[f"p0_{k}"], R[f"target_{k}"] = p0[k], tgt[k].cpu().numpy()

    # ---------------- 실험 A (clean 모델): 배경 개입
    pA = np.zeros((n, nB + 2 * nD), np.float32)
    mA = np.zeros_like(pA)
    for i in range(n):
        xp, keep = xs[i].to(dev), bird[i:i + 1].to(dev)  # keep = 새 (배경을 바꿈)
        ims = [fill(b, xp, keep, pix_mean, fills[i].to(dev), fa) for b in BASELINES]
        for d in list(same[i]) + list(opp[i]):
            ims.append(xp * keep + (1 - keep) * bank_pix[d].to(dev))
        batch = torch.cat([im.reshape(-1, 3, 224, 224) for im in ims])
        _, p = nets["clean"](batch)
        pA[i] = p[:, tgt["clean"][i]].cpu().numpy()
        mA[i] = scorer.maha(refnet(batch)[0].cpu().numpy())
    R["A_p"], R["A_ref_maha"] = pA, mA  # 열: baseline 5 | same 4 | opp 4

    # ---------------- H3-2 단계 1: 칸별 자연 교체
    cells = [cell_mask(c, dev) for c in range(GRID * GRID)]
    maxdp = {k: np.zeros((n, GRID * GRID), np.float32) for k in nets}
    for i in range(n):
        xp = xs[i].to(dev)
        donors = [bank_pix[d].to(dev) for d in list(same[i]) + list(opp[i])]  # 물 4 + 땅 4
        batch = torch.cat([xp[None] * (1 - m) + dn[None] * m for m in cells for dn in donors])  # (16*8,3,H,W)
        for k, net in nets.items():
            _, p = net(batch)
            dp = (p0[k][i] - p[:, tgt[k][i]].cpu().numpy()).reshape(GRID * GRID, len(donors))
            maxdp[k][i] = np.abs(dp).max(1)
    for k in nets:
        R[f"S1_maxdp_{k}"] = maxdp[k]

    # ---------------- H3-2 단계 2: 공통 안 보는 칸 마스킹
    for tau in TAUS:
        common = (maxdp["clean"] < tau) & (maxdp["shortcut"] < tau)  # (n,16)
        R[f"S2_common_tau{tau}"] = common
        pS = {k: np.full((n, nB + nD), np.nan, np.float32) for k in nets}
        cS = {k: np.full((n, nB + nD), np.nan, np.float32) for k in nets}
        mS = np.full((n, nB + nD), np.nan, np.float32)
        for i in range(n):
            if not common[i].any():
                continue
            cm = torch.clamp(sum(cells[c] for c in np.where(common[i])[0]), 0, 1)
            xp, keep = xs[i].to(dev), 1 - cm
            ims = [fill(b, xp, keep, pix_mean, fills[i].to(dev), fa) for b in BASELINES]
            ims += [xp * keep + cm * bank_pix[d].to(dev) for d in same[i]]
            batch = torch.cat([im.reshape(-1, 3, 224, 224) for im in ims])
            for k, net in nets.items():
                f, p = net(batch)
                pS[k][i] = p[:, tgt[k][i]].cpu().numpy()
                cS[k][i] = F.cosine_similarity(f, f0[k][i:i + 1]).cpu().numpy()
            mS[i] = scorer.maha(refnet(batch)[0].cpu().numpy())
        for k in nets:
            R[f"S2_p_{k}_tau{tau}"], R[f"S2_cos_{k}_tau{tau}"] = pS[k], cS[k]
        R[f"S2_ref_maha_tau{tau}"] = mS
        print(f"[{args.arch} seed{args.seed}] tau={tau}: images with >=2 common cells "
              f"{(common.sum(1) >= 2).mean():.2f}, mean common cells {common.sum(1).mean():.2f}", flush=True)

    np.savez_compressed(out / "raw.npz", img_filename=df.img_filename.values, group=(df.y * 2 + df.place).values,
                        place=df.place.values, bird_area=bird.mean((1, 2, 3)).numpy(), baselines=np.array(BASELINES),
                        **R)
    cfg = dict(arch=args.arch, seed=args.seed, runs=runs, n_images=n, limit=args.limit, taus=TAUS, grid=GRID,
               n_donor_per_place=nD, columns_A="baseline x5 | same-place x4 | opposite-place x4",
               columns_S2="baseline x5 | same-place natural x4", ref="ImageNet ResNet-50, single-Gaussian Mahalanobis",
               seconds=round(time.time() - t0, 1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"[{args.arch} seed{args.seed}] done ({cfg['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
