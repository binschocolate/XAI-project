"""ERM 학습 (고정 epoch, 마지막 checkpoint 사용).

사용: python src/train.py --arch resnet18 --rho 0.95 --seed 0 --data_dir $DATA_DIR
결과: <out_dir>/<run_name>/{config.json, train_log.csv, final.pt}
"""
import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import GROUPS, WaterbirdsDataset, build_model, get_transform, rho_tag, run_name, set_seed


def seed_worker(worker_id, base=0):
    np.random.seed(base * 1000 + worker_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "resnet50"], required=True)
    ap.add_argument("--rho", type=float, choices=[0.5, 0.95], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--out_dir", default=f"/data/{os.environ.get('USER', 'user')}/results/XAI-project/models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--hflip", action="store_true", help="좌우 반전 augmentation (기본 off)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    name = run_name(args.arch, args.rho, args.seed)
    run_dir = Path(args.out_dir) / name
    if (run_dir / "final.pt").exists() and not args.overwrite:
        print(f"[skip] {run_dir}/final.pt already exists")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(Path(args.split_dir) / f"train_rho{rho_tag(args.rho)}_seed{args.seed}.csv")
    ds = WaterbirdsDataset(train_df, args.data_dir, get_transform(hflip=args.hflip))
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        worker_init_fn=partial(seed_worker, base=args.seed),
    )

    model = build_model(args.arch).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    config = {**vars(args), "run_name": name, "n_train": len(ds), "device": device}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(json.dumps(config, ensure_ascii=False))

    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        correct, count = np.zeros(4), np.zeros(4)
        tot_loss = 0.0
        for x, y, place, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * len(y)
            hit = (out.argmax(1) == y).cpu().numpy()
            gid = (y.cpu().numpy() * 2 + place.numpy())  # (y,place) -> 0..3
            np.add.at(correct, gid, hit)
            np.add.at(count, gid, 1)

        row = {"epoch": epoch, "loss": tot_loss / count.sum(), "acc": correct.sum() / count.sum(),
               "sec": round(time.time() - t0, 1)}
        for k, gname in GROUPS.items():
            j = k[0] * 2 + k[1]
            row[f"acc_{gname}"] = correct[j] / count[j] if count[j] else float("nan")
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    torch.save(model.state_dict(), run_dir / "final.pt")
    print(f"[done] saved {run_dir}/final.pt")


if __name__ == "__main__":
    main()
