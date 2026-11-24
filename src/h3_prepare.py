"""H3 실험 준비 (CPU, 1회). 설계: docs/H3_DESIGN.md

만드는 것 (outputs/h3/prep/)
  test_masks.npz   test_xai800 800장의 새 mask (800,224,224) uint8 (1 = 새), 새 면적 비율, 장소
  bank.npz         자연 배경 은행: train+val pool 에서 장소별 새 면적이 가장 작은 64장 × 2, 새 영역을 push-pull inpaint 로 지운
                   "새 없는 배경" (128,3,224,224) uint8 + 장소 + 파일명
  donors.npz       test 이미지마다 은행에서 고른 같은 장소 4장 / 반대 장소 4장 (seed 0 고정)
  report.txt       segmentation 정렬 확인 (크기 불일치 수 등)
  oracle_attr/<run>/{bird,background,random}.npy + meta.csv   (clean 6개; 실험 B 용 정답을 아는 설명)
      bird = 새 mask, background = 1 − 새 mask (+ 동점 순서용 1e-6 크기 고정 난수; _shared/ 에 한 번 저장 후 링크), random = 단계 5 random map 링크

전처리: 모델 입력과 같게 Resize(256×256) → CenterCrop(224), mask 는 nearest. segmentation 값 ≥ 128 = 새.
사용: python src/h3_prepare.py --data_dir $DATA_DIR --seg_dir data/segmentations
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from common import get_transform
from faithfulness import push_pull_inpaint

REPO = Path(__file__).resolve().parent.parent
MASK_T = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(224)])
IMG_T = get_transform()  # 정규화 포함 → 아래에서 픽셀로 되돌림
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLEAN_RUNS = [f"resnet{a}_rho050_seed{s}" for a in (18, 50) for s in range(3)]


def load_mask(seg_dir, data_dir, fname, stats):
    seg = Image.open(seg_dir / fname.replace(".jpg", ".png")).convert("L")
    w, h = Image.open(data_dir / fname).size
    if seg.size != (w, h):  # Waterbirds 이미지와 크기가 다르면 맞춤 (보고서에 개수 기록)
        stats["resized"] += 1
        seg = seg.resize((w, h), Image.NEAREST)
    m = np.array(MASK_T(seg)) >= 128
    return m.astype(np.uint8)


def load_pix(data_dir, fname):
    return IMG_T(Image.open(data_dir / fname).convert("RGB")) * STD + MEAN  # (3,224,224) [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seg_dir", default=str(REPO / "data" / "segmentations"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "h3" / "prep"))
    ap.add_argument("--bank_per_place", type=int, default=64)
    ap.add_argument("--n_donor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data_dir, seg_dir, out = Path(args.data_dir), Path(args.seg_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"resized": 0}

    # 1) test 800장 새 mask
    test = pd.read_csv(REPO / "splits" / "test_xai800.csv")
    masks = np.stack([load_mask(seg_dir, data_dir, f, stats) for f in test.img_filename])
    area = masks.reshape(len(test), -1).mean(1)
    np.savez_compressed(out / "test_masks.npz", bird=masks, area=area, place=test.place.values,
                        img_filename=test.img_filename.values)
    print(f"test masks: {masks.shape}, bird area mean {area.mean():.3f} (min {area.min():.3f}, max {area.max():.3f})", flush=True)

    # 2) 자연 배경 은행 (train+val pool, 장소별 새 면적 하위 64장)
    meta = pd.read_csv(data_dir / "metadata.csv")
    pool = meta[meta.split.isin([0, 1])].reset_index(drop=True)
    pool_area = np.array([load_mask(seg_dir, data_dir, f, stats).mean() for f in pool.img_filename])
    pick = np.concatenate([pool.index[pool.place == p].values[np.argsort(pool_area[pool.place.values == p], kind="stable")
                                                               [:args.bank_per_place]] for p in (0, 1)])
    bank, bank_place = [], []
    for i in pick:
        f = pool.img_filename[i]
        x = load_pix(data_dir, f)
        bird = torch.from_numpy(load_mask(seg_dir, data_dir, f, stats)).float()[None, None]
        clean = push_pull_inpaint(x[None], 1 - bird, x.mean((1, 2)).view(3, 1, 1))[0]  # 새 자리를 주변 배경으로 채움
        bank.append((clean.clamp(0, 1) * 255).round().byte().numpy())
        bank_place.append(int(pool.place[i]))
    bank = np.stack(bank)
    bank_place = np.array(bank_place)
    np.savez_compressed(out / "bank.npz", pix=bank, place=bank_place, img_filename=pool.img_filename.values[pick],
                        bird_area=pool_area[pick])
    print(f"bank: {bank.shape}, bird area max per place "
          f"{[round(float(pool_area[pick][bank_place == p].max()), 3) for p in (0, 1)]}", flush=True)

    # 3) test 이미지별 donor (같은 장소 4 / 반대 장소 4)
    rng = np.random.default_rng(args.seed)
    same, opp = [], []
    for p in test.place.values:
        same.append(rng.choice(np.where(bank_place == p)[0], args.n_donor, replace=False))
        opp.append(rng.choice(np.where(bank_place != p)[0], args.n_donor, replace=False))
    np.savez_compressed(out / "donors.npz", same=np.stack(same), opp=np.stack(opp))

    # 4) oracle map (clean 6개, 실험 B)
    noise = np.random.default_rng(args.seed + 1).random(masks.shape).astype(np.float32) * 1e-6
    shared = out.parent / "oracle_attr" / "_shared"  # 모델과 무관하므로 한 번만 저장하고 run 폴더에는 링크
    shared.mkdir(parents=True, exist_ok=True)
    np.save(shared / "bird.npy", masks.astype(np.float32) + noise)
    np.save(shared / "background.npy", (1 - masks).astype(np.float32) + noise)
    for run in CLEAN_RUNS:
        src = REPO / "outputs" / "attributions" / run
        if not (src / "meta.csv").exists():
            print(f"[skip oracle] {run}: 단계 5 결과 없음")
            continue
        d = out.parent / "oracle_attr" / run
        d.mkdir(parents=True, exist_ok=True)
        am = pd.read_csv(src / "meta.csv")
        assert (am.img_filename.values == test.img_filename.values).all()
        for name, target in [("bird.npy", shared / "bird.npy"), ("background.npy", shared / "background.npy"),
                             ("random.npy", src / "random.npy")]:
            (d / name).unlink(missing_ok=True)
            (d / name).symlink_to(target)
        am.to_csv(d / "meta.csv", index=False)
    rep = dict(n_test=len(test), n_pool=len(pool), seg_resized=stats["resized"], test_bird_area_mean=float(area.mean()),
               bank_per_place=args.bank_per_place, n_donor=args.n_donor, seed=args.seed)
    (out / "report.txt").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
