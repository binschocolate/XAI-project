"""공통 test set(5,794장) 평가 + SRI 계산.

SRI (Shortcut Reliance Index)
  = 클래스별 |평균 P(waterbird | water 배경) - 평균 P(waterbird | land 배경)| 의 평균
  test set은 클래스 안에서 배경이 50:50 무작위 배정이므로, 이 차이는 배경의 효과로 해석한다.

사용: python src/eval_sri.py --run_dir /data/$USER/results/XAI-project/models/resnet18_rho095_seed0 --data_dir $DATA_DIR
결과: <run_dir>/{test_preds.csv, metrics.json}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ALIGNED, CONFLICTING, GROUPS, WaterbirdsDataset, build_model, get_transform


def compute_metrics(t: pd.DataFrame) -> dict:
    """t: columns y, place, p_wb, pred"""
    t = t.assign(correct=(t.pred == t.y).astype(float))
    m = {"n_test": len(t), "avg_acc": t.correct.mean()}
    gacc = {}
    for k, gname in GROUPS.items():
        sel = t[(t.y == k[0]) & (t.place == k[1])]
        gacc[k] = sel.correct.mean()
        m[f"acc_{gname}"] = gacc[k]
        m[f"p_wb_{gname}"] = sel.p_wb.mean()
    m["worst_group_acc"] = min(gacc.values())
    m["group_gap"] = np.mean([gacc[k] for k in ALIGNED]) - np.mean([gacc[k] for k in CONFLICTING])

    sri_cls = []
    for y in (0, 1):
        pw = t[(t.y == y) & (t.place == 1)].p_wb.mean()
        pl = t[(t.y == y) & (t.place == 0)].p_wb.mean()
        sri_cls.append(abs(pw - pl))
    m["sri_landbird"], m["sri_waterbird"] = sri_cls
    m["sri"] = float(np.mean(sri_cls))
    return {k: float(v) if not isinstance(v, int) else v for k, v in m.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _, _, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg["arch"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location="cpu", weights_only=True))
    model.to(device)

    test = pd.read_csv(Path(args.split_dir) / "test.csv")
    loader = DataLoader(WaterbirdsDataset(test, args.data_dir, get_transform()),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    p = predict(model, loader, device)
    out = test.assign(p_wb=p, pred=(p > 0.5).astype(int))
    out.to_csv(run_dir / "test_preds.csv", index=False)

    metrics = {"run_name": cfg["run_name"], "arch": cfg["arch"], "rho": cfg["rho"], "seed": cfg["seed"],
               **compute_metrics(out)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
