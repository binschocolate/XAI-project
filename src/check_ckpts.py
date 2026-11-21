"""[ariel 전용] 팀원에게 받은 checkpoint(final.pt)를 불러와 test 평가 -> git 의 팀 결과와 대조.

확인하는 것
  1. checkpoint 가 팀 build_model(arch) 구조로 문제없이 로드되는지
  2. 우리 서버 데이터(WILDS 배포본, ceph)로 낸 예측이 팀원 Seraph 예측(results/<arch>/runs/<run>/test_preds.csv)과 같은지
     -> p_wb 최대 차이, 예측 일치율. 거의 0 / 1.0 이면 checkpoint 와 데이터 모두 팀과 동일하다는 뜻
  3. avg acc / worst-group / group gap / SRI 가 팀 metrics.json 과 같은지

checkpoint 위치: <ckpt_root>/<run>/final.pt   (기본 ckpt_root = repo/models, 예: models/resnet50_rho095_seed0/final.pt)
config.json 은 ckpt 폴더에 없으면 git 의 results/<arch>/runs/<run>/config.json 을 사용.

사용: python src/check_ckpts.py --data_dir $DATA_DIR                 # ckpt_root 아래 전부
      python src/check_ckpts.py --data_dir $DATA_DIR --runs resnet18_rho095_seed0
결과: <ckpt_root>/<run>/{test_preds_ariel.csv, metrics_ariel.json}, <out_csv>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import WaterbirdsDataset, build_model, get_transform
from eval_sri import compute_metrics, predict

REPO = Path(__file__).resolve().parent.parent
KEYS = ["avg_acc", "worst_group_acc", "group_gap", "sri"]


def team_dir(run):
    return REPO / "results" / run.split("_")[0] / "runs" / run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt_root", default=str(REPO / "models"))
    ap.add_argument("--runs", nargs="*", default=None, help="생략하면 ckpt_root 안의 */final.pt 전부")
    ap.add_argument("--split_dir", default=str(REPO / "splits"))
    ap.add_argument("--out_csv", default=str(REPO / "outputs" / "ckpt_check.csv"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    runs = args.runs or sorted(p.parent.name for p in root.glob("*/final.pt"))
    if not runs:
        raise SystemExit(f"final.pt 없음: {root}/<run>/final.pt 에 두세요")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} runs={runs}", flush=True)

    test = pd.read_csv(Path(args.split_dir) / "test.csv")
    loader = DataLoader(WaterbirdsDataset(test, args.data_dir, get_transform()),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows = []
    for run in runs:
        rd, td = root / run, team_dir(run)
        cfg_path = rd / "config.json" if (rd / "config.json").exists() else td / "config.json"
        cfg = json.loads(cfg_path.read_text())

        model = build_model(cfg["arch"], pretrained=False)
        model.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True))  # 키 불일치면 여기서 에러
        model.to(device)

        p = predict(model, loader, device)
        out = test.assign(p_wb=p, pred=(p > 0.5).astype(int))
        out.to_csv(rd / "test_preds_ariel.csv", index=False)
        m = compute_metrics(out)
        (rd / "metrics_ariel.json").write_text(json.dumps(m, indent=2))

        row = {"run": run, "arch": cfg["arch"], "rho": cfg["rho"], "seed": cfg["seed"],
               **{f"ariel_{k}": m[k] for k in KEYS}}
        if (td / "metrics.json").exists():
            tm = json.loads((td / "metrics.json").read_text())
            row.update({f"team_{k}": tm[k] for k in KEYS})
            row["max_abs_metric_diff"] = max(abs(m[k] - tm[k]) for k in KEYS)
        if (td / "test_preds.csv").exists():
            tp = pd.read_csv(td / "test_preds.csv")
            assert (tp.img_filename.values == out.img_filename.values).all(), "test 순서 불일치"
            row["pred_agreement"] = float((tp.pred.values == out.pred.values).mean())
            row["max_abs_p_wb_diff"] = float(np.abs(tp.p_wb.values - out.p_wb.values).max())
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("\n" + df.round(4).to_string(index=False))
    if "pred_agreement" in df:
        print("\n판정 기준: pred_agreement ≈ 1.0 & max_abs_p_wb_diff < ~1e-3 이면 팀과 동일한 모델·데이터")


if __name__ == "__main__":
    main()
