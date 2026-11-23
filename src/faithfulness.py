"""단계 6: Faithfulness 평가 — patch 를 중요도 순서대로 지우며 모델 점수 곡선을 잰다.

입력: outputs/attributions/<run>/{meta.csv, gradcam/ig/rise/random.npy}  (단계 5 결과)
출력: outputs/faithfulness/<run>/
  auc.csv       이미지 × method × baseline 한 줄씩: mif_auc, lif_auc, insertion_auc, srg, p_orig, p_full_baseline
  curves.npz    mif / lif 곡선 원본 (n_img, n_method, n_baseline, n_point), x = 지운 비율
  summary.csv   method × baseline × {all, aligned, conflicting} 평균
  config.json   설정

정의 (팀 PROJECT_PLAN 단계 6, Blücher et al. 2024)
  - 제거 단위: 14×14 grid patch (224px 이미지 -> 16×16px patch 196개). patch 중요도 = 그 안 attribution 평균
  - 점수: target class 의 softmax 확률 (target = 단계 5 meta.csv 의 target, 기본은 모델 예측 class)
  - MIF (= Deletion): 중요한 patch 부터 지움. AUC 낮을수록 좋음
  - LIF: 덜 중요한 patch 부터 지움. AUC 높을수록 좋음
  - Insertion: baseline 이미지에 중요한 patch 부터 복원. "중요 k개만 남은 상태" = "LIF 로 나머지를 지운 상태" 라서
               곡선이 LIF 를 뒤집은 것과 같고 AUC 도 같다 -> 따로 계산하지 않고 insertion_auc = lif_auc 로 기록
  - SRG = LIF AUC − MIF AUC (클수록 좋음)
  - method=random 은 무작위 순서 곡선 (= R-OMS 용 대조군, baseline 자체의 OOD 효과 측정)
  - AUC: x = 지운 비율 [0, 1] 위의 사다리꼴 적분

Baseline (지운 patch 를 무엇으로 채우나, 모두 정규화 전 픽셀 공간에서 채운 뒤 정규화)
  black     픽셀 0
  mean      Waterbirds train+val pool 의 채널별 평균 픽셀 (처음 1회 계산 후 캐시)
  blur      원본 이미지의 Gaussian blur (kernel 41, sigma 8)
  inpaint   남은 픽셀로부터 multi-scale 확산(push-pull) 으로 채움. GPU 에서 빠르게 돌아가는 고전 inpainting.
            (Diffusion inpainting 은 팀 계획상 시간 부족 시 제외 대상이라 쓰지 않음)
  trainset  train+val pool 에서 이미지마다 고정된 무작위 다른 이미지 1장의 같은 위치 픽셀

사용:
  python src/faithfulness.py --data_dir $DATA_DIR --runs resnet18_rho095_seed0
  python src/faithfulness.py --data_dir $DATA_DIR --runs resnet18_rho095_seed0 --limit 16 --out_root .../faithfulness_smoke
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
from torch.utils.data import DataLoader

from attribution import SIZE, load_model
from common import IMAGENET_MEAN, IMAGENET_STD, WaterbirdsDataset, get_transform, set_seed

REPO = Path(__file__).resolve().parent.parent
METHODS = ["gradcam", "ig", "rise", "random"]
BASELINES = ["black", "mean", "blur", "inpaint", "trainset"]


# ----------------------------------------------------------------------------- baselines
def push_pull_inpaint(xp, keep, fallback):
    """xp: (B,3,H,W) 픽셀, keep: (B,1,H,W) 1=남김. 지운 곳을 주변에서 multi-scale 로 채움.
    전부 지워진 경우 fallback (3,1,1) 색으로 채움."""
    xs, ws = [xp * keep], [keep]
    while ws[-1].shape[-1] > 1:
        xs.append(F.avg_pool2d(xs[-1], 2, ceil_mode=True))
        ws.append(F.avg_pool2d(ws[-1], 2, ceil_mode=True))
    w = ws[-1]
    fill = torch.where(w > 0, xs[-1] / w.clamp_min(1e-8), fallback.expand_as(xs[-1]))
    for xl, wl in zip(reversed(xs[:-1]), reversed(ws[:-1])):
        up = F.interpolate(fill, size=xl.shape[-2:], mode="bilinear", align_corners=False)
        fill = xl + (1 - wl) * up  # 알려진 비율만큼은 실제 값, 나머지는 위 단계에서 보간
    return fill


def train_pixel_mean(data_dir, cache, num_workers):
    if cache.exists():
        return torch.tensor(json.loads(cache.read_text())["mean"]).view(3, 1, 1)
    meta = pd.read_csv(Path(data_dir) / "metadata.csv")
    pool = meta[meta.split.isin([0, 1])]
    dl = DataLoader(WaterbirdsDataset(pool, data_dir, get_transform()), batch_size=128, num_workers=num_workers)
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    s, n = torch.zeros(3, dtype=torch.float64), 0
    for x, _, _, _ in dl:
        xp = x * std + mean
        s += xp.sum(dim=(0, 2, 3)).double()
        n += xp.shape[0] * xp.shape[2] * xp.shape[3]
    m = (s / n).float()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mean": m.tolist(), "n_images": len(pool), "space": "pixel [0,1], Resize256-CenterCrop224"}))
    return m.view(3, 1, 1)


def load_trainset_fills(data_dir, n, seed, num_workers):
    """이미지 i 마다 train+val pool 의 고정된 무작위 이미지 1장 (모든 모델 공통). (n,3,224,224) 픽셀"""
    meta = pd.read_csv(Path(data_dir) / "metadata.csv")
    pool = meta[meta.split.isin([0, 1])].reset_index(drop=True)
    pick = pool.iloc[np.random.default_rng(seed).choice(len(pool), n, replace=False)]
    dl = DataLoader(WaterbirdsDataset(pick, data_dir, get_transform()), batch_size=64, num_workers=num_workers)
    x = torch.cat([b for b, _, _, _ in dl])
    return x * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), \
        pick.img_filename.tolist()


# ----------------------------------------------------------------------------- core
def patch_orders(attr, grid):
    """attr: (H,W) -> (MIF 순서, LIF 순서) patch index 배열. patch 점수 = 평균 attribution."""
    a = torch.from_numpy(np.array(attr, dtype=np.float32))[None, None]  # mmap 은 읽기 전용이라 복사
    s = F.adaptive_avg_pool2d(a, grid).flatten()
    mif = torch.argsort(s, descending=True, stable=True)
    return mif, mif.flip(0)


def keep_masks(orders, ks, grid, device):
    """orders: list of (n_patch,) 순서. ks: 지울 patch 수 목록 -> (len(orders)*len(ks), 1, H, W) 1=남김"""
    n_patch = grid * grid
    rank = torch.empty(len(orders), n_patch, dtype=torch.long)
    for j, o in enumerate(orders):
        rank[j, o] = torch.arange(n_patch)  # rank[j, p] = patch p 가 몇 번째로 지워지나
    ks_t = torch.as_tensor(ks)
    keep = (rank[:, None, :] >= ks_t[None, :, None]).float()  # (n_order, n_k, n_patch)
    keep = keep.view(-1, 1, grid, grid).to(device)
    return F.interpolate(keep, size=(SIZE, SIZE), mode="nearest")


@torch.no_grad()
def score(model, inp, target, chunk, amp):
    out = []
    for i in range(0, inp.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and inp.is_cuda):
            logits = model(inp[i:i + chunk])
        out.append(F.softmax(logits.float(), 1)[:, target])
    return torch.cat(out)


def auc(curve, x):
    trap = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x 는 trapezoid
    return trap(curve, x, axis=-1)


def run_one(run, args, xs_pix, meta, fills_train, pix_mean, device, mean, std):
    out = Path(args.out_root) / run
    if (out / "auc.csv").exists() and not args.overwrite:
        print(f"[{run}] auc.csv exists, skip (--overwrite 로 다시)", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = load_model(run, args.ckpt_root, device)
    adir = Path(args.attr_root) / run
    ameta = pd.read_csv(adir / "meta.csv")
    missing = set(meta.img_filename) - set(ameta.img_filename)
    assert not missing, f"단계 5 결과에 없는 이미지 {len(missing)}장"
    # 단계 5 배열에서의 행 번호는 파일명으로 찾는다 (attribution 을 --limit 으로 일부만 만든 경우에도 맞도록)
    row = pd.Index(ameta.img_filename).get_indexer(meta.img_filename)
    ameta = ameta.iloc[row].reset_index(drop=True)  # 이 스크립트의 이미지 순서로 정렬
    attrs = {m: np.load(adir / f"{m}.npy", mmap_mode="r") for m in args.methods}
    a_idx = meta["attr_idx"].values  # test_xai800 기준 이미지 번호 (출력 기록용)

    grid, n_patch = args.grid, args.grid * args.grid
    ks = list(range(0, n_patch + 1, args.step_patches))
    if ks[-1] != n_patch:
        ks.append(n_patch)
    x = np.array(ks) / n_patch
    nI, nM, nB, nP = len(meta), len(args.methods), len(args.baselines), len(ks)
    mif = np.zeros((nI, nM, nB, nP), np.float32)
    lif = np.zeros((nI, nM, nB, nP), np.float32)
    p_full = np.zeros((nI, nB), np.float32)

    for i in range(nI):
        xp = xs_pix[i].to(device)
        target = int(ameta.target[i])
        orders = []
        for m in args.methods:
            o_mif, o_lif = patch_orders(attrs[m][row[i]], grid)
            orders += [o_mif, o_lif]
        keep = keep_masks(orders, ks, grid, device)  # (nM*2*nP, 1, H, W)
        for b, base in enumerate(args.baselines):
            if base == "black":
                filled = xp * keep
            elif base == "mean":
                filled = xp * keep + (1 - keep) * pix_mean
            elif base == "blur":
                blurred = TF.gaussian_blur(xp[None], kernel_size=args.blur_kernel, sigma=args.blur_sigma)
                filled = xp * keep + (1 - keep) * blurred
            elif base == "trainset":
                filled = xp * keep + (1 - keep) * fills_train[i].to(device)
            elif base == "inpaint":
                filled = torch.cat([push_pull_inpaint(xp[None].expand(c.shape[0], -1, -1, -1), c, pix_mean)
                                    for c in keep.split(args.chunk)])
            p = score(model, (filled - mean) / std, target, args.chunk, not args.no_amp).cpu().numpy()
            p = p.reshape(nM, 2, nP)
            mif[i, :, b], lif[i, :, b] = p[:, 0], p[:, 1]
            p_full[i, b] = p[0, 0, -1]  # 전부 지운 이미지 (= baseline 만 남은 이미지) 의 점수
        if (i + 1) % 100 == 0 or i + 1 == nI:
            print(f"[{run}] {i + 1}/{nI} images ({time.time() - t0:.0f}s)", flush=True)

    mif_auc, lif_auc = auc(mif, x), auc(lif, x)
    rows = []
    for i in range(nI):
        for mi, m in enumerate(args.methods):
            for b, base in enumerate(args.baselines):
                rows.append(dict(img_idx=int(a_idx[i]), img_filename=meta.img_filename[i], y=int(meta.y[i]),
                                 place=int(meta.place[i]), group=int(meta.group[i]),
                                 aligned=int(meta.group[i] in (0, 3)), target=int(ameta.target[i]),
                                 method=m, baseline=base, mif_auc=mif_auc[i, mi, b], lif_auc=lif_auc[i, mi, b],
                                 insertion_auc=lif_auc[i, mi, b], srg=lif_auc[i, mi, b] - mif_auc[i, mi, b],
                                 p_orig=mif[i, mi, b, 0], p_full_baseline=p_full[i, b]))
    df = pd.DataFrame(rows)
    df.insert(0, "run", run)
    df.to_csv(out / "auc.csv", index=False)
    np.savez_compressed(out / "curves.npz", mif=mif, lif=lif, x=x, methods=np.array(args.methods),
                        baselines=np.array(args.baselines), img_idx=a_idx)

    cols = ["mif_auc", "lif_auc", "insertion_auc", "srg", "p_full_baseline"]
    parts = [df.assign(subset="all"), df[df.aligned == 1].assign(subset="aligned"),
             df[df.aligned == 0].assign(subset="conflicting")]
    summ = pd.concat(parts).groupby(["subset", "method", "baseline"])[cols].mean().reset_index()
    summ.insert(0, "run", run)
    summ.to_csv(out / "summary.csv", index=False)

    cfg = dict(run=run, n_images=nI, methods=args.methods, baselines=args.baselines, grid=grid,
               patch_px=SIZE // grid, step_patches=args.step_patches, n_points=nP, score="softmax prob of target",
               target_source=str(adir / "meta.csv"), blur={"kernel": args.blur_kernel, "sigma": args.blur_sigma},
               mean_pixel=pix_mean.flatten().tolist(), trainset_seed=args.trainset_seed,
               amp_fp16=not args.no_amp, seconds=round(time.time() - t0, 1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"[{run}] done ({cfg['seconds']}s)\n" +
          summ[summ.subset == "all"].pivot(index="method", columns="baseline", values="srg").round(3).to_string(),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="생략하면 attr_root 아래 전부")
    ap.add_argument("--ckpt_root", default=str(REPO / "models"))
    ap.add_argument("--attr_root", default=str(REPO / "outputs" / "attributions"))
    ap.add_argument("--out_root", default=str(REPO / "outputs" / "faithfulness"))
    ap.add_argument("--split", default=str(REPO / "splits" / "test_xai800.csv"))
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    help="attr_root/<run>/<method>.npy 가 있는 이름 (기본 4종; H3 실험 B 는 bird background random)")
    ap.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    ap.add_argument("--grid", type=int, default=14, help="patch grid (14 -> 14×14 = 196 patch, 16px)")
    ap.add_argument("--step_patches", type=int, default=4, help="곡선 한 점마다 지우는 patch 수")
    ap.add_argument("--blur_kernel", type=int, default=41)
    ap.add_argument("--blur_sigma", type=float, default=8.0)
    ap.add_argument("--trainset_seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200, help="forward batch 크기")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="그룹별 limit/4 장만 (동작 확인용, 결과 해석 금지)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = args.runs or sorted(p.parent.name for p in Path(args.attr_root).glob("*/meta.csv"))
    print(f"device={device} runs={runs}\nmethods={args.methods} baselines={args.baselines}", flush=True)

    df = pd.read_csv(args.split)
    df["attr_idx"] = np.arange(len(df))
    if args.limit:
        df = df.groupby(["y", "place"], group_keys=False).head(max(1, args.limit // 4)).reset_index(drop=True)
    meta = df[["img_filename", "y", "place", "attr_idx"]].assign(group=df.y * 2 + df.place)
    dl = DataLoader(WaterbirdsDataset(df, args.data_dir, get_transform()), batch_size=64, num_workers=args.num_workers)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    xs_pix = torch.cat([x for x, _, _, _ in dl]) * std.cpu() + mean.cpu()
    print(f"loaded {len(xs_pix)} images", flush=True)

    pix_mean = train_pixel_mean(args.data_dir, REPO / "outputs" / "faithfulness" / "train_pixel_mean.json",
                                args.num_workers).to(device)
    fills_train = None
    if "trainset" in args.baselines:
        # attr_idx 기준으로 고정 -> --limit 여부와 무관하게 같은 이미지가 같은 채움 이미지를 받음
        all_fills, _ = load_trainset_fills(args.data_dir, len(pd.read_csv(args.split)),
                                           args.trainset_seed, args.num_workers)
        fills_train = all_fills[torch.as_tensor(df.attr_idx.values)]
    print(f"train pixel mean = {pix_mean.flatten().tolist()}", flush=True)

    for run in runs:
        run_one(run, args, xs_pix, meta, fills_train, pix_mean, device, mean, std)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
