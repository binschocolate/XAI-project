"""ROAR (RemOve And Retrain; Hooker et al. 2019) — 지운 이미지로 모델을 다시 학습해 평가.

Deletion(MIF)은 지운 이미지를 원래 모델에 바로 넣으므로 "마스킹이 낯설어서" 생기는 효과가 섞인다.
ROAR 는 같은 방식으로 지운 학습 이미지로 **처음부터 다시 학습**한 뒤, 지운 test 이미지의 정확도를 잰다.
중요한 곳을 잘 찾은 설명일수록 정확도가 더 많이 떨어진다 (Random 순서보다 낮아야 의미 있음).

설정 (2026-11-23 결정)
  - 모델(run): ResNet-18·50 × clean/shortcut × seed 0–2 (12개). 설명은 그 run 의 원래 모델(models/<run>/final.pt)로 만든 것
  - 학습 데이터: 그 run 의 B안 split 1,678장 / 평가: splits/test_xai800.csv 800장 (다른 지표와 같은 subset)
  - 설명: Grad-CAM, IG, RISE, Random (단계 5와 같은 설정·target = 원래 모델의 예측 class)
      · test 800장은 단계 5 결과(outputs/attributions/<run>/) 재사용
      · 학습 1,678장은 src/attribution.py --split <train csv> 로 따로 생성 (outputs/roar/attr_train/<run>/)
  - 지우는 단위·순서: 14×14 grid(196 patch), patch 평균 attribution 내림차순으로 상위 10/30/50/70% 제거
  - 채우는 값: mean (train+val 평균색, ROAR 원 논문 방식) / trainset (이미지마다 고정된 다른 학습 이미지의 같은 위치)
      · test 의 trainset donor 는 단계 6과 동일, 학습 이미지의 donor 는 같은 split 의 다른 이미지(seed 고정)
  - 재학습: 팀 학습과 동일 — ImageNet 사전학습에서 시작, SGD lr 1e-3, momentum 0.9, wd 1e-4, batch 32, 30 epoch,
    augmentation 없음, 마지막 checkpoint. seed = run 의 seed (모든 조건이 같은 초기화·배치 순서)
  - 지우기 0% (마스킹 없이 재학습) 1회를 기준선으로 포함
  - 조건 수: run 당 1 + 4 method × 4 비율 × 2 baseline = 33회 재학습
출력: <out_root>/<run>/roar.csv (조건별 test 정확도, 그룹별 정확도, 학습 정확도), config.json
     중단 후 다시 실행하면 이미 끝난 조건은 건너뜀
사용: python src/roar.py --data_dir $DATA_DIR --run resnet18_rho095_seed0
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
from torch.utils.data import DataLoader

from attribution import SIZE
from common import IMAGENET_MEAN, IMAGENET_STD, WaterbirdsDataset, build_model, get_transform, set_seed
from faithfulness import load_trainset_fills, train_pixel_mean

REPO = Path(__file__).resolve().parent.parent
METHODS = ["gradcam", "ig", "rise", "random"]


def load_pixels(df, data_dir, num_workers):
    """(n,3,224,224) float16 픽셀 [0,1] (CPU 메모리)"""
    dl = DataLoader(WaterbirdsDataset(df, data_dir, get_transform()), batch_size=128, num_workers=num_workers)
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return torch.cat([(x * std + mean).half() for x, *_ in dl])


def patch_ranks(attr_dir, method, files, grid):
    """attribution map -> 이미지별 patch 제거 순위 (n,196): 0 = 가장 먼저 지움 (중요도 내림차순)"""
    meta = pd.read_csv(attr_dir / "meta.csv")
    row = pd.Index(meta.img_filename).get_indexer(files)
    assert (row >= 0).all(), f"attribution 에 없는 이미지: {attr_dir}"
    maps = np.load(attr_dir / f"{method}.npy", mmap_mode="r")
    ranks = np.empty((len(files), grid * grid), np.int16)
    for i, r in enumerate(row):
        s = F.adaptive_avg_pool2d(torch.from_numpy(np.array(maps[r], dtype=np.float32))[None, None], grid).flatten()
        order = torch.argsort(s, descending=True, stable=True)
        ranks[i, order.numpy()] = np.arange(grid * grid)
    return torch.from_numpy(ranks)


def masked_batch(xp, ranks, k, grid, baseline, pix_mean, donor):
    """xp (b,3,H,W) 픽셀, ranks (b,196) -> 상위 k patch 를 지운 픽셀 이미지"""
    if k == 0:
        return xp
    keep = (ranks >= k).float().view(-1, 1, grid, grid)
    keep = F.interpolate(keep, size=(SIZE, SIZE), mode="nearest")
    fill = pix_mean.expand_as(xp) if baseline == "mean" else donor
    return xp * keep + (1 - keep) * fill


def run_condition(arch, seed, data, method, baseline, frac, args, device, mean, std, pix_mean):
    grid = args.grid
    k = int(round(frac * grid * grid))
    tr, te = data["train"], data["test"]
    set_seed(seed)
    model = build_model(arch, pretrained=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    n = len(tr["y"])
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        correct = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xp = tr["x"][idx].to(device).float()
            rk = tr["ranks"][method][idx].to(device) if k else None
            xp = masked_batch(xp, rk, k, grid, baseline, pix_mean, tr["donor"][idx].to(device).float() if k else None)
            y = tr["y"][idx].to(device)
            out = model((xp - mean) / std)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
    train_acc = correct / n
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te["y"]), 128):
            sl = slice(i, i + 128)
            xp = te["x"][sl].to(device).float()
            rk = te["ranks"][method][sl].to(device) if k else None
            xp = masked_batch(xp, rk, k, grid, baseline, pix_mean, te["donor"][sl].to(device).float() if k else None)
            preds.append(model((xp - mean) / std).argmax(1).cpu())
    ok = (torch.cat(preds) == te["y"]).numpy()
    gr = te["group"]
    res = dict(test_acc=ok.mean(), aligned_acc=ok[np.isin(gr, [0, 3])].mean(),
               conflicting_acc=ok[np.isin(gr, [1, 2])].mean(), train_acc_last_epoch=train_acc)
    res.update({f"acc_g{j}": ok[gr == j].mean() for j in range(4)})
    res["worst_group_acc"] = min(res[f"acc_g{j}"] for j in range(4))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--attr_train_root", default=str(REPO / "outputs" / "roar" / "attr_train"))
    ap.add_argument("--attr_test_root", default=str(REPO / "outputs" / "attributions"))
    ap.add_argument("--out_root", default=str(REPO / "outputs" / "roar"))
    ap.add_argument("--split", default=str(REPO / "splits" / "test_xai800.csv"))
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--baselines", nargs="+", default=["mean", "trainset"], choices=["mean", "trainset"])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--donor_seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_train", type=int, default=None, help="동작 확인용: 학습 이미지 수 제한 (결과 해석 금지)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    arch, rho, seed_tag = args.run.split("_")
    seed = int(seed_tag.replace("seed", ""))
    out = Path(args.out_root) / args.run
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "roar.csv"
    done = pd.read_csv(csv) if csv.exists() and not args.overwrite else pd.DataFrame()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    pix_mean = train_pixel_mean(args.data_dir, REPO / "outputs" / "faithfulness" / "train_pixel_mean.json",
                                args.num_workers).to(device).view(1, 3, 1, 1)
    t0 = time.time()

    # 데이터 (메모리에 한 번만 올림)
    tr_df = pd.read_csv(REPO / "splits" / f"train_{rho}_{seed_tag}.csv")
    if args.limit_train:  # attribution.py --limit 과 같은 선택 (그룹별 limit/4 장)
        tr_df = tr_df.groupby(["y", "place"], group_keys=False).head(max(1, args.limit_train // 4)).reset_index(drop=True)
    te_df = pd.read_csv(args.split)
    data = {"train": {"x": load_pixels(tr_df, args.data_dir, args.num_workers), "y": torch.as_tensor(tr_df.y.values)},
            "test": {"x": load_pixels(te_df, args.data_dir, args.num_workers), "y": torch.as_tensor(te_df.y.values),
                     "group": (te_df.y * 2 + te_df.place).values}}
    rng = np.random.default_rng(args.donor_seed + seed)
    n_tr = len(tr_df)
    donor_idx = (np.arange(n_tr) + rng.integers(1, n_tr, n_tr)) % n_tr  # 자기 자신이 아닌 다른 학습 이미지
    data["train"]["donor"] = data["train"]["x"][torch.as_tensor(donor_idx)]
    fills, _ = load_trainset_fills(args.data_dir, len(te_df), 0, args.num_workers)  # 단계 6과 같은 test donor
    data["test"]["donor"] = fills.half()
    for part, root, df in [("train", Path(args.attr_train_root), tr_df), ("test", Path(args.attr_test_root), te_df)]:
        data[part]["ranks"] = {m: patch_ranks(root / args.run, m, df.img_filename.values, args.grid) for m in args.methods}
    print(f"[{args.run}] data ready: train {n_tr}, test {len(te_df)} ({time.time() - t0:.0f}s)", flush=True)

    conds = [("none", "none", 0.0)] + [(m, b, f) for b in args.baselines for m in args.methods for f in args.fractions]
    rows = done.to_dict("records") if len(done) else []
    for method, baseline, frac in conds:
        if len(done) and ((done.method == method) & (done.baseline == baseline) & (done.fraction == frac)).any():
            continue
        t1 = time.time()
        res = run_condition(arch, seed, data, method if method != "none" else METHODS[0], baseline, frac,
                            args, device, mean, std, pix_mean)
        rows.append(dict(run=args.run, arch=arch, model="shortcut" if rho == "rho095" else "clean", seed=seed,
                         method=method, baseline=baseline, fraction=frac, **res, sec=round(time.time() - t1, 1)))
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{args.run}] {method:8s} {baseline:8s} {frac:.1f}  test_acc {res['test_acc']:.4f}  "
              f"train_acc {res['train_acc_last_epoch']:.3f}  ({time.time() - t1:.0f}s)", flush=True)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update(arch=arch, seed=seed, n_train=n_tr, n_test=len(te_df), n_conditions=len(conds),
               init="ImageNet pretrained (team build_model)", train_donor="same split, fixed per image (donor_seed + seed)",
               test_donor="stage-6 trainset fills (seed 0)", seconds_total=round(time.time() - t0, 1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"[{args.run}] all done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
