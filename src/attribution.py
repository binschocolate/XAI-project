"""단계 5: Attribution 생성 (Grad-CAM, Integrated Gradients, RISE, Random).

모델(run) 하나씩 불러와 splits/test_xai800.csv (그룹당 200장) 에 대해 attribution map 을 만들고 저장한다.
단계 6 (Deletion/Insertion 등) 은 이 파일들을 읽기만 한다.

저장 (<out_root>/<run>/)
  meta.csv          행 순서 = test_xai800.csv 순서. img_filename, y, place, group, p_wb, pred, target
  <method>.npy      float32 (N, 224, 224). 모델 입력(Resize 256 -> CenterCrop 224) 좌표계 기준 픽셀 단위 map
                    (14×14 patch 집계는 단계 6 에서 함. 값이 클수록 target class 에 중요)
  config.json       설정 + method 별 소요 시간, IG completeness 오차
  vis.png           (--vis) 그룹별 2장씩 시각 확인용

기본 설정 (팀 PROJECT_PLAN: "IG baseline / RISE fill 은 기본값으로 고정")
  target   : 모델의 예측 class (--target label 이면 정답 class)
  Grad-CAM : 마지막 conv 블록(layer4) 출력, target logit 기준, ReLU, bilinear 로 224 업샘플
  IG       : baseline = 정규화 전 픽셀 공간의 black(0), 50 step (midpoint Riemann), target logit 기준,
             채널 합(부호 유지)
  RISE     : 마스크 4000개, 7×7 grid, p=0.5, 가려진 곳은 black(0) 픽셀, target softmax 확률 기준.
             마스크는 seed 고정이라 모든 모델·이미지가 같은 마스크를 씀
  Random   : 이미지마다 고정 seed 의 균일 난수 map (모든 모델 공통) — 대조군

사용:
  python src/attribution.py --data_dir $DATA_DIR --runs resnet18_rho095_seed0 --vis
  python src/attribution.py --data_dir $DATA_DIR --runs resnet18_rho095_seed0 --limit 16 --rise_n 500   # 빠른 확인
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import IMAGENET_MEAN, IMAGENET_STD, WaterbirdsDataset, build_model, get_transform, set_seed

REPO = Path(__file__).resolve().parent.parent
ALL_METHODS = ["gradcam", "ig", "rise", "random"]
SIZE = 224


# ----------------------------------------------------------------------------- methods
def gradcam(model, x, target):
    """x: (B,3,H,W) 정규화된 입력, target: (B,) -> (B,H,W)"""
    store = {}

    def hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = model.layer4.register_forward_hook(hook)
    try:
        x = x.clone().requires_grad_(True)  # 파라미터는 고정, 입력 쪽으로 그래프를 만들어 hook 이 동작하게
        score = model(x).gather(1, target[:, None]).sum()
        score.backward()
    finally:
        h.remove()
    a, g = store["a"].detach(), store["g"].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam[:, 0]


def integrated_gradients(model, x, target, baseline, steps=50, chunk=50):
    """x: (3,H,W) 한 장, target: int -> (map (H,W), completeness 상대오차)"""
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    grad_sum = torch.zeros_like(x)
    for i in range(0, steps, chunk):
        a = alphas[i:i + chunk].view(-1, 1, 1, 1)
        inp = (baseline + a * (x - baseline)).requires_grad_(True)
        out = model(inp)[:, target].sum()
        (g,) = torch.autograd.grad(out, inp)
        grad_sum += g.sum(0)
    attr = ((x - baseline) * grad_sum / steps).sum(0)
    with torch.no_grad():
        f = model(torch.stack([x, baseline]))[:, target]
        delta = (f[0] - f[1]).item()
    err = abs(attr.sum().item() - delta) / (abs(delta) + 1e-8)
    return attr, err


def make_rise_masks(n, s, p, seed):
    """Petsiuk et al. 2018: s×s 이진 grid -> bilinear 업샘플 -> 랜덤 shift crop. (n,1,224,224) CPU"""
    g = torch.Generator().manual_seed(seed)
    cell = int(np.ceil(SIZE / s))
    up = (s + 1) * cell
    grid = (torch.rand(n, 1, s, s, generator=g) < p).float()
    shifts = torch.randint(0, cell, (n, 2), generator=g)
    masks = torch.empty(n, 1, SIZE, SIZE)
    for i in range(0, n, 256):
        upm = F.interpolate(grid[i:i + 256], size=(up, up), mode="bilinear", align_corners=False)
        for j in range(upm.shape[0]):
            dx, dy = shifts[i + j].tolist()
            masks[i + j] = upm[j, :, dx:dx + SIZE, dy:dy + SIZE]
    return masks


@torch.no_grad()
def rise(model, x, target, masks, p, mean, std, batch=400, amp=True):
    """x: (3,H,W) 정규화된 입력. 가려진 곳 = 픽셀 black(0). -> (H,W)"""
    xp = x * std + mean  # 정규화 해제 (픽셀 공간)
    sal = torch.zeros(SIZE * SIZE, device=x.device)
    for i in range(0, masks.shape[0], batch):
        m = masks[i:i + batch]
        inp = (xp * m - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and x.is_cuda):
            logits = model(inp)
        prob = F.softmax(logits.float(), 1)[:, target]
        sal += prob @ m.view(m.shape[0], -1)
    return (sal / (masks.shape[0] * p)).view(SIZE, SIZE)


# ----------------------------------------------------------------------------- utils
def load_model(run, ckpt_root, device):
    model = build_model(run.split("_")[0], pretrained=False)
    model.load_state_dict(torch.load(Path(ckpt_root) / run / "final.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    model.requires_grad_(False)  # 파라미터 gradient 불필요 (입력 gradient 만 사용)
    return model


def save_vis(path, xs, meta, maps, mean, std):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = [i for g in range(4) for i in np.where(meta.group.values == g)[0][:2]]
    names = list(maps)
    fig, axes = plt.subplots(len(idx), 1 + len(names), figsize=(2.2 * (1 + len(names)), 2.2 * len(idx)))
    gname = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
    for r, i in enumerate(idx):
        img = (xs[i] * std.cpu() + mean.cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{gname[meta.group[i]]}\np_wb={meta.p_wb[i]:.2f} t={meta.target[i]}", fontsize=7)
        for c, name in enumerate(names, start=1):
            m = np.asarray(maps[name][i], dtype=np.float32)
            if name == "ig":  # 픽셀 단위 + 부호 -> 절댓값을 살짝 blur 하고 99% 에서 자름
                m = np.abs(m)
                m = F.avg_pool2d(torch.from_numpy(m)[None, None], 9, 1, 4)[0, 0].numpy()
            lo, hi = np.percentile(m, 1), np.percentile(m, 99)  # RISE 는 공통 offset 이 커서 min-max 필요
            m = np.clip((m - lo) / (hi - lo + 1e-12), 0, 1)
            axes[r, c].imshow(img)
            axes[r, c].imshow(m, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(name, fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run_one(run, args, xs, base_meta, device, mean, std, masks):
    out = Path(args.out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(run, args.ckpt_root, device)
    n = len(xs)

    # 원본 예측과 target
    with torch.no_grad():
        logits = torch.cat([model(xs[i:i + args.batch_size].to(device)).float().cpu()
                            for i in range(0, n, args.batch_size)])
    p_wb = F.softmax(logits, 1)[:, 1].numpy()
    pred = logits.argmax(1).numpy()
    meta = base_meta.assign(p_wb=p_wb, pred=pred)
    meta["target"] = pred if args.target == "pred" else meta.y.values
    meta.to_csv(out / "meta.csv", index=False)
    target = torch.as_tensor(meta.target.values, device=device)
    print(f"[{run}] acc on xai subset = {(pred == meta.y.values).mean():.4f}", flush=True)

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() and not args.overwrite else {}
    prev_ig_err = cfg.get("ig", {}).get("completeness_rel_err_median")  # ig 를 skip 할 때 기존 값 유지
    cfg.update(run=run, n_images=n, split=str(args.split), target=args.target, input="Resize256-CenterCrop224",
               ig={"baseline": "pixel black (0)", "steps": args.ig_steps},
               rise={"n_masks": args.rise_n, "grid": args.rise_s, "p": args.rise_p, "fill": "pixel black (0)",
                     "mask_seed": args.mask_seed, "amp_fp16": not args.no_amp},
               gradcam={"layer": "layer4", "score": "logit"}, random={"seed": "image index"})
    cfg.setdefault("timing_sec", {})
    if prev_ig_err is not None:
        cfg["ig"]["completeness_rel_err_median"] = prev_ig_err

    maps = {}
    for method in args.methods:
        f = out / f"{method}.npy"
        if f.exists() and not args.overwrite:
            print(f"[{run}] {method}: exists, skip (--overwrite 로 다시 생성)", flush=True)
            maps[method] = np.load(f, mmap_mode="r")
            continue
        t0 = time.time()
        res = np.zeros((n, SIZE, SIZE), dtype=np.float32)
        if method == "gradcam":
            for i in range(0, n, args.batch_size):
                res[i:i + args.batch_size] = gradcam(model, xs[i:i + args.batch_size].to(device),
                                                     target[i:i + args.batch_size]).cpu().numpy()
        elif method == "ig":
            baseline = ((torch.zeros(3, 1, 1) - mean.cpu()) / std.cpu()).expand(3, SIZE, SIZE).to(device)
            errs = []
            for i in range(n):
                a, e = integrated_gradients(model, xs[i].to(device), int(target[i]), baseline, args.ig_steps)
                res[i] = a.cpu().numpy()
                errs.append(e)
            cfg["ig"]["completeness_rel_err_median"] = float(np.median(errs))
            print(f"[{run}] ig completeness 상대오차 median={np.median(errs):.4f} (작을수록 step 충분)", flush=True)
        elif method == "rise":
            for i in range(n):
                res[i] = rise(model, xs[i].to(device), int(target[i]), masks, args.rise_p, mean, std,
                              args.rise_batch, amp=not args.no_amp).cpu().numpy()
                if (i + 1) % 100 == 0:
                    print(f"[{run}] rise {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        elif method == "random":
            for i in range(n):
                res[i] = np.random.default_rng(args.random_seed + i).random((SIZE, SIZE), dtype=np.float32)
        np.save(f, res)
        maps[method] = res
        cfg["timing_sec"][method] = round(time.time() - t0, 1)
        print(f"[{run}] {method}: saved {f} ({cfg['timing_sec'][method]}s)", flush=True)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    if args.vis:
        save_vis(out / "vis.png", xs, meta, maps, mean, std)
        print(f"[{run}] vis -> {out / 'vis.png'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="생략하면 ckpt_root 아래 */final.pt 전부")
    ap.add_argument("--ckpt_root", default=str(REPO / "models"))
    ap.add_argument("--split", default=str(REPO / "splits" / "test_xai800.csv"))
    ap.add_argument("--out_root", default=str(REPO / "outputs" / "attributions"))
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--target", choices=["pred", "label"], default="pred")
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--rise_n", type=int, default=4000)
    ap.add_argument("--rise_s", type=int, default=7)
    ap.add_argument("--rise_p", type=float, default=0.5)
    ap.add_argument("--rise_batch", type=int, default=400)
    ap.add_argument("--mask_seed", type=int, default=0)
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true", help="RISE forward 를 fp32 로 (느림)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="앞에서부터 N장만 (동작 확인용, 결과 해석 금지)")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = args.runs or sorted(p.parent.name for p in Path(args.ckpt_root).glob("*/final.pt"))
    if not runs:
        raise SystemExit(f"final.pt 없음: {args.ckpt_root}")
    print(f"device={device} runs={runs} methods={args.methods}", flush=True)

    df = pd.read_csv(args.split)
    if args.limit:
        df = df.groupby(["y", "place"], group_keys=False).head(max(1, args.limit // 4)).reset_index(drop=True)
    base_meta = df[["img_filename", "y", "place"]].assign(group=df.y * 2 + df.place)
    loader = DataLoader(WaterbirdsDataset(df, args.data_dir, get_transform()), batch_size=64,
                        shuffle=False, num_workers=args.num_workers)
    xs = torch.cat([x for x, _, _, _ in loader])  # (N,3,224,224) 정규화된 입력, 800장 ≈ 0.5GB
    print(f"loaded {len(xs)} images from {args.split}", flush=True)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    masks = make_rise_masks(args.rise_n, args.rise_s, args.rise_p, args.mask_seed).to(device) \
        if "rise" in args.methods else None

    for run in runs:
        run_one(run, args, xs, base_meta, device, mean, std, masks)
    print("done", flush=True)


if __name__ == "__main__":
    main()
