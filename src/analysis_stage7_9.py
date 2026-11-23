"""단계 7–9 분석 (docs/PROJECT_PLAN.md "단계 7–10 상세 실행 계획" 명세 구현). CPU 전용.

입력
  outputs/faithfulness/<run>/{auc.csv, curves.npz, config.json}     (단계 6)
  outputs/attributions/<run>/{meta.csv, config.json}                (단계 5)
  models/<run>/metrics_ariel.json, results/<arch>/runs/<run>/metrics.json   (SRI: ariel 재평가 / 팀 Seraph)
출력 (outputs/analysis/)
  validation_report.md        7-0 입력 완전성 검사
  faith_auc.csv               7-1 run × method × baseline × scope 평균 MIF/LIF/SRG, 표본 수, MRG
  roms.csv                    7-1/9-2 Random 순서 AUC, p_orig, p_full_baseline
  score_variation.csv         7-2 baseline 5종에 대한 평균 점수 SD(ddof=0)·Range
  sensitivity.csv             7-2 순위 민감도 S = 1 − mean Kendall τ-b, FlipRate, 유효 쌍 수, 순위표
  method_pair_differences.csv 7-3 방법 쌍 d(a,a',b) + paired bootstrap 95% 구간
  paired_comparison.csv       8-1 같은 arch·seed 의 D = S_shortcut − S_clean
  bootstrap_summary.csv       8-3/8-4/9-1 D, arch 평균 D, G, A, I 의 이미지 bootstrap 95% 구간
  group_comparison.csv        9-1 scope 별 D 와 I = D_conflicting − D_aligned
  robustness.csv              9-4 Black 제외 / baseline 하나씩 제외 / 공통 정답 이미지
  figs/*.png                  paired plot, SRI–S 산점도, Random 곡선

정의 (명세 그대로)
  - 순위 대상: Grad-CAM, IG, RISE (Random 은 순위에 넣지 않음). Q = −MIF 또는 SRG (클수록 좋음)
  - scope 평균: scope 안 그룹별 평균을 동일 가중 (800장 기본 subset 에서는 이미지 평균과 같음)
  - Kendall τ-b: 정확한 동점은 공동 순위. τ 정의 불가(한쪽이 전부 동점)한 쌍은 제외하고 유효 쌍 수 보고
  - FlipRate: 공동 순위 관계(쌍별 부호)가 다른 baseline 쌍 비율
  - bootstrap: 4그룹 각 200장 복원추출, 2,000회, seed 0, 같은 추출을 모든 run·method·baseline 에 적용,
    매 반복 평균 AUC → 순위 → S → D 를 다시 계산. 구간 = 고정 모델에 조건부인 평가 이미지 불확실성
사용: python src/analysis_stage7_9.py [--n_boot 2000]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
METHODS = ["gradcam", "ig", "rise", "random"]
XAI = ["gradcam", "ig", "rise"]
BASES = ["black", "mean", "blur", "inpaint", "trainset"]
GNAME = ["landbird_land", "landbird_water", "waterbird_land", "waterbird_water"]
SCOPES = {"all": [0, 1, 2, 3], "aligned": [0, 3], "conflicting": [1, 2],
          **{f"g{k}_{GNAME[k]}": [k] for k in range(4)}}
QS = ["mif", "srg"]  # 순위 민감도 지표 (명세: MIF 주 분석, SRG 주요 보조)
METHOD_PAIRS = [("gradcam", "ig"), ("gradcam", "rise"), ("ig", "rise")]
trap = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------- 순위 민감도 (벡터화 τ-b)
def rank_sensitivity(Q, bidx=None):
    """Q: (..., 3 method, B baseline) 클수록 좋은 점수 -> S, FlipRate, 유효 쌍 수, 전체 쌍 수"""
    if bidx is not None:
        Q = Q[..., bidx]
    s = np.stack([np.sign(Q[..., i, :] - Q[..., j, :]) for i, j in [(0, 1), (0, 2), (1, 2)]], -2)  # (...,3,B)
    taus, flips = [], []
    for b1, b2 in combinations(range(Q.shape[-1]), 2):
        num = (s[..., b1] * s[..., b2]).sum(-1)
        den = np.sqrt((s[..., b1] != 0).sum(-1) * (s[..., b2] != 0).sum(-1))
        taus.append(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))
        flips.append((s[..., b1] != s[..., b2]).any(-1))
    taus, flips = np.stack(taus, -1), np.stack(flips, -1)
    nvalid = (~np.isnan(taus)).sum(-1)
    with np.errstate(invalid="ignore"):
        S = 1 - np.where(nvalid > 0, np.nansum(taus, -1) / np.maximum(nvalid, 1), np.nan)
    return S, flips.mean(-1), nvalid, taus.shape[-1]


def _selftest():
    """작은 인공 입력으로 τ-b·순위 방향·동점 처리를 scipy 와 대조"""
    from scipy.stats import kendalltau
    rng = np.random.default_rng(0)
    for _ in range(300):
        Q = rng.integers(0, 3, size=(3, 5)).astype(float)  # 동점 많이
        S, _, nv, _ = rank_sensitivity(Q)
        ts = [kendalltau(Q[:, a], Q[:, b])[0] for a, b in combinations(range(5), 2)]
        ts = [t for t in ts if not np.isnan(t)]
        ref = 1 - np.mean(ts) if ts else np.nan
        assert (np.isnan(S) and np.isnan(ref)) or abs(S - ref) < 1e-9, (Q, S, ref)
        assert nv == len(ts)
    same = np.tile(np.array([[3.0], [2.0], [1.0]]), (1, 5))
    assert rank_sensitivity(same)[0] == 0 and rank_sensitivity(same)[1] == 0
    assert rank_sensitivity(-same)[0] == 0  # 방향을 뒤집어도 τ 는 같음 (−MIF vs MIF)


# ----------------------------------------------------------------------------- 로딩 + 7-0 검증
def parse_run(run):
    arch, rho, seed = run.split("_")
    return arch, ("shortcut" if rho == "rho095" else "clean"), int(seed[-1])


def load_all(froot, aroot, report, n_expected=12, sri_root=None, fixed_order=False):
    runs = sorted(p.parent.name for p in froot.glob("*/auc.csv"))
    ok = lambda c, msg: report.append(("PASS" if c else "FAIL", msg))
    ok(len(runs) == n_expected, f"run 수 = {len(runs)} (기대 {n_expected})")
    ref = pd.read_csv(REPO / "splits" / "test_xai800.csv")
    data = {}
    for run in runs:
        am = pd.read_csv(aroot / run / "meta.csv")
        df = pd.read_csv(froot / run / "auc.csv")
        cur = np.load(froot / run / "curves.npz")
        # 1. 동일 800장, 그룹별 200장 (파일명으로 대응)
        ok((am.img_filename.values == ref.img_filename.values).all(), f"[{run}] attribution meta 순서 = test_xai800.csv")
        ok(sorted(am.group.value_counts().tolist()) == [200] * 4, f"[{run}] 그룹별 200장")
        # 3·4. 키 중복·누락, 행 수
        key = ["img_filename", "method", "baseline"]
        ok(not df.duplicated(key).any(), f"[{run}] (img, method, baseline) 중복 없음")
        ok(len(df) == 800 * 4 * 5, f"[{run}] auc.csv 행 수 {len(df)} (= 800×4×5; MIF/LIF 는 열 → long 형식 32,000행에 해당)")
        idx = pd.MultiIndex.from_product([am.img_filename, METHODS, BASES], names=key)
        w = df.set_index(key).reindex(idx)
        ok(not w[["mif_auc", "lif_auc"]].isna().any().any(), f"[{run}] method×baseline 누락 없음")
        mif = w.mif_auc.values.reshape(800, 4, 5)
        lif = w.lif_auc.values.reshape(800, 4, 5)
        # 2. 설명 대상 = 원본 예측 class, attribution 과 평가가 같은 target
        tgt = w.target.values.reshape(800, 4, 5)
        ok((tgt == am.target.values[:, None, None]).all(), f"[{run}] 평가 target = attribution target")
        ok((am.target == am.pred).all(), f"[{run}] target = 원본 예측 class (고정)")
        # 5. 범위
        for name, a, lo in [("MIF", mif, 0), ("LIF", lif, 0), ("SRG", lif - mif, -1)]:
            ok(np.isfinite(a).all() and a.min() >= lo - 1e-6 and a.max() <= 1 + 1e-6,
               f"[{run}] {name} 유한·범위 [{a.min():.3f}, {a.max():.3f}]")
        # 6. 곡선 시작점 = 원본 확률, 끝점은 method 간 동일
        order = np.argsort(cur["img_idx"])
        cm, cl = cur["mif"][order], cur["lif"][order]
        assert list(cur["methods"]) == METHODS and list(cur["baselines"]) == BASES
        p_t = np.where(am.target == 1, am.p_wb, 1 - am.p_wb).astype(np.float32)
        d0 = np.abs(cm[..., 0] - p_t[:, None, None]).max()
        if fixed_order:  # 다른 모델의 attribution·target 으로 평가 → 시작점이 그 모델 확률과 다른 것이 정상
            report.append(("INFO", f"[{run}] 고정 순서 진단: 곡선 시작점 vs attribution 모델 확률 최대 차 {d0:.4f} (검사 제외)"))
        else:
            ok(d0 < 0.02, f"[{run}] 곡선 시작점 vs 원본 예측 확률 최대 차 {d0:.4f} (평가는 fp16 autocast, 허용 0.02)")
        dend = max(np.abs(cm[..., -1] - cm[:, :1, :, -1]).max(), np.abs(cl[..., -1] - cm[:, :1, :, -1]).max())
        ok(dend < 0.02, f"[{run}] 완전 삭제 끝점 method·MIF/LIF 간 최대 차 {dend:.4f} (배치 구성별 fp16 차이, 허용 0.02)")
        # 7. CSV AUC = 곡선 재적분
        rm = np.abs(trap(cm, cur["x"], axis=-1) - mif).max()
        ok(rm < 1e-4, f"[{run}] CSV AUC vs 곡선 재적분 최대 차 {rm:.2e} (곡선 float32 저장)")
        # 9. SRI
        sri_path = (Path(sri_root) / run / "metrics.json") if sri_root else (REPO / "models" / run / "metrics_ariel.json")
        sri_a = json.loads(sri_path.read_text())["sri"]
        tm = REPO / "results" / run.split("_")[0] / "runs" / run / "metrics.json"
        sri_t = json.loads(tm.read_text())["sri"] if tm.exists() else np.nan
        ok(np.isfinite(sri_a), f"[{run}] SRI ariel={sri_a:.4f} team={sri_t:.4f}")
        data[run] = dict(mif=mif, lif=lif, g=am.group.values, y=am.y.values, pred=am.pred.values,
                         p_orig=w.p_orig.values.reshape(800, 4, 5), p_full=w.p_full_baseline.values.reshape(800, 4, 5),
                         sri=sri_a, sri_team=sri_t, curves=(cm, cl, cur["x"]))
    # run 간 이미지 대응
    files = [tuple(pd.read_csv(aroot / r / "meta.csv").img_filename) for r in runs]
    ok(len(set(files)) == 1, f"{len(runs)}개 run 의 이미지 목록·순서 동일 (img_filename 기준)")
    return runs, data


def group_mean(A, g, mask=None):
    """A: (n, ...) -> (4, ...) 그룹 평균 (mask 로 선택된 이미지만). 빈 그룹은 NaN"""
    out = []
    for k in range(4):
        sel = (g == k) if mask is None else (g == k) & mask
        out.append(A[sel].mean(0) if sel.any() else np.full(A.shape[1:], np.nan))
    return np.stack(out)


def scope_mean(GM, scope):
    return GM[SCOPES[scope]].mean(0)


def qarr(d, q):
    """클수록 좋은 Q: −MIF 또는 SRG (n, 4 method, 5 base)"""
    return -d["mif"] if q == "mif" else d["lif"] - d["mif"]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--froot", default=str(REPO / "outputs" / "faithfulness"))
    ap.add_argument("--aroot", default=str(REPO / "outputs" / "attributions"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "analysis"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_runs", type=int, default=12, help="기대 run 수 (추가 학습 실험은 6)")
    ap.add_argument("--sri_root", default=None, help="<sri_root>/<run>/metrics.json 에서 SRI 읽기 (기본: models/<run>/metrics_ariel.json)")
    ap.add_argument("--fixed_order", action="store_true", help="다른 모델의 attribution 으로 평가한 고정 순서 진단 결과")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    _selftest()
    report = [("PASS", "τ-b 벡터화 구현 = scipy.stats.kendalltau (인공 입력 300개, 동점 포함)")]
    runs, data = load_all(Path(args.froot), Path(args.aroot), report, args.n_runs, args.sri_root, args.fixed_order)
    fails = [m for s, m in report if s == "FAIL"]
    (out / "validation_report.md").write_text(
        "# 단계 7-0 입력 검증\n\n" + "\n".join(f"- {s}: {m}" for s, m in report) +
        f"\n\n**결과: {'모두 통과' if not fails else f'실패 {len(fails)}건'}**\n")
    print(f"[7-0] checks={len(report)} fail={len(fails)}")
    if fails:
        raise SystemExit("검증 실패 — validation_report.md 확인")
    meta = {r: dict(zip(["arch", "model", "seed"], parse_run(r)), sri=data[r]["sri"], sri_team=data[r]["sri_team"])
            for r in runs}
    xi = [METHODS.index(m) for m in XAI]
    ri = METHODS.index("random")

    # ---------------- 7-1 faith_auc / roms
    rows, rrows = [], []
    for r in runs:
        d = data[r]
        GM = {k: group_mean(d[k], d["g"]) for k in ["mif", "lif", "p_orig", "p_full"]}
        for sc in SCOPES:
            m_, l_ = scope_mean(GM["mif"], sc), scope_mean(GM["lif"], sc)
            n = int(np.isin(d["g"], SCOPES[sc]).sum())
            for mi, m in enumerate(METHODS):
                for bi, b in enumerate(BASES):
                    rows.append(dict(run=r, **meta[r], method=m, baseline=b, scope=sc, n=n, mif=m_[mi, bi],
                                     lif=l_[mi, bi], srg=l_[mi, bi] - m_[mi, bi],
                                     mrg=m_[ri, bi] - m_[mi, bi]))
            for bi, b in enumerate(BASES):
                rrows.append(dict(run=r, **meta[r], baseline=b, scope=sc, n=n, random_mif=m_[ri, bi],
                                  random_lif=l_[ri, bi], random_srg=l_[ri, bi] - m_[ri, bi],
                                  p_orig=scope_mean(GM["p_orig"], sc)[0, bi],
                                  p_full_baseline=scope_mean(GM["p_full"], sc)[0, bi]))
    fa = pd.DataFrame(rows)
    fa.to_csv(out / "faith_auc.csv", index=False)
    pd.DataFrame(rrows).to_csv(out / "roms.csv", index=False)

    # ---------------- 7-2 score variation / sensitivity (점추정)
    sv = fa.melt(id_vars=["run", "arch", "model", "seed", "sri", "method", "baseline", "scope"],
                 value_vars=["mif", "lif", "srg"], var_name="metric")
    sv = sv.groupby(["run", "arch", "model", "seed", "sri", "method", "scope", "metric"]).value.agg(
        sd=lambda v: np.std(v, ddof=0), range=lambda v: v.max() - v.min(), n_baselines="count").reset_index()
    sv.to_csv(out / "score_variation.csv", index=False)

    def S_point(r, q, sc, bidx=None, mask=None):
        Q = qarr(data[r], q)[:, xi]
        return rank_sensitivity(scope_mean(group_mean(Q, data[r]["g"], mask), sc), bidx)

    srows = []
    for r in runs:
        for q in QS:
            for sc in SCOPES:
                Qm = scope_mean(group_mean(qarr(data[r], q)[:, xi], data[r]["g"]), sc)
                S, fr, nv, npair = rank_sensitivity(Qm)
                order = {b: ">".join(np.array(XAI)[np.argsort(-Qm[:, bi], kind="stable")]) for bi, b in enumerate(BASES)}
                srows.append(dict(run=r, **meta[r], metric=q, scope=sc, S=S, flip_rate=fr, n_valid_pairs=int(nv),
                                  n_pairs=npair, **{f"rank_{b}": o for b, o in order.items()}))
    sens = pd.DataFrame(srows)
    sens.to_csv(out / "sensitivity.csv", index=False)

    # ---------------- bootstrap 인덱스 (그룹별 200장 복원추출, 모든 run 공통)
    rng = np.random.default_rng(args.seed)
    g0 = data[runs[0]]["g"]
    pos = [np.where(g0 == k)[0] for k in range(4)]
    bidx = [p[rng.integers(0, len(p), size=(args.n_boot, len(p)))] for p in pos]  # 4 × (R, 200)

    def boot_GM(A):
        """A: (n, ...) -> (R, 4, ...) 반복별 그룹 평균"""
        return np.stack([A[b].mean(1) for b in bidx], 1)

    # S 의 bootstrap 분포: Sb[run][q][scope] (R,)
    Sb, Sb_drop = {}, {}
    for r in runs:
        Sb[r] = {}
        for q in QS:
            GMb = boot_GM(qarr(data[r], q)[:, xi])  # (R,4,3,5)
            Sb[r][q] = {sc: rank_sensitivity(GMb[:, SCOPES[sc]].mean(1))[0] for sc in SCOPES}

    # ---------------- 7-3 방법 쌍 차이 (paired bootstrap)
    prow = []
    for r in runs:
        for q in QS:
            Q = qarr(data[r], q)
            for a, a2 in METHOD_PAIRS:
                diff = Q[:, METHODS.index(a)] - Q[:, METHODS.index(a2)]  # (n, 5)
                pt = scope_mean(group_mean(diff, data[r]["g"]), "all")
                bt = boot_GM(diff).mean(1)  # (R,5)
                lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
                for bi, b in enumerate(BASES):
                    prow.append(dict(run=r, **meta[r], metric=q, scope="all", pair=f"{a}-{a2}", baseline=b,
                                     d=pt[bi], ci_lo=lo[bi], ci_hi=hi[bi], ci_excludes_0=bool(lo[bi] > 0 or hi[bi] < 0)))
    mp = pd.DataFrame(prow)
    mp["sign_changes_across_baselines"] = mp.groupby(["run", "metric", "pair"]).d.transform(
        lambda v: (np.sign(v) > 0).any() and (np.sign(v) < 0).any())
    mp.to_csv(out / "method_pair_differences.csv", index=False)

    # ---------------- 8-1 / 9-1 paired D, I + 8-3/8-4 bootstrap
    archs = sorted({meta[r]["arch"] for r in runs})
    run_of = {(meta[r]["arch"], meta[r]["model"], meta[r]["seed"]): r for r in runs}
    pc, bs, gc = [], [], []
    pct = lambda v: np.nanpercentile(v, [2.5, 97.5])
    for arch in archs:
        for q in QS:
            for sc in SCOPES:
                Ds, Dbs = [], []
                for seed in range(3):
                    rs, rc = run_of[(arch, "shortcut", seed)], run_of[(arch, "clean", seed)]
                    s_s = sens.query("run==@rs and metric==@q and scope==@sc").iloc[0]
                    s_c = sens.query("run==@rc and metric==@q and scope==@sc").iloc[0]
                    D = s_s.S - s_c.S
                    Db = Sb[rs][q][sc] - Sb[rc][q][sc]
                    Ds.append(D)
                    Dbs.append(Db)
                    pc.append(dict(arch=arch, seed=seed, metric=q, scope=sc, S_clean=s_c.S, S_shortcut=s_s.S, D=D,
                                   flip_clean=s_c.flip_rate, flip_shortcut=s_s.flip_rate,
                                   nvalid_clean=s_c.n_valid_pairs, nvalid_shortcut=s_s.n_valid_pairs,
                                   SRI_clean=meta[rc]["sri"], SRI_shortcut=meta[rs]["sri"]))
                    lo, hi = pct(Db)
                    bs.append(dict(quantity="D", arch=arch, seed=seed, metric=q, scope=sc, estimate=D, ci_lo=lo, ci_hi=hi))
                Dm = np.nanmean(np.stack(Dbs), 0)
                lo, hi = pct(Dm)
                bs.append(dict(quantity="D_arch_mean", arch=arch, seed="mean", metric=q, scope=sc,
                               estimate=np.mean(Ds), ci_lo=lo, ci_hi=hi, n_seeds_D_pos=int(np.sum(np.array(Ds) > 0)),
                               sd_seeds=float(np.std(Ds, ddof=1))))
    pc = pd.DataFrame(pc)
    pc.to_csv(out / "paired_comparison.csv", index=False)

    # 8-4 G = S_MIF − S_SRG (모델별), A = D_MIF − D_SRG (arch·seed 별)  /  9-1 I = D_conf − D_aligned
    for r in runs:
        for sc in ["all", "aligned", "conflicting"]:
            G = sens.query("run==@r and metric=='mif' and scope==@sc").S.iloc[0] - \
                sens.query("run==@r and metric=='srg' and scope==@sc").S.iloc[0]
            lo, hi = pct(Sb[r]["mif"][sc] - Sb[r]["srg"][sc])
            bs.append(dict(quantity="G", arch=meta[r]["arch"], seed=meta[r]["seed"], model=meta[r]["model"],
                           metric="mif-srg", scope=sc, estimate=G, ci_lo=lo, ci_hi=hi))
    for arch in archs:
        for seed in range(3):
            rs, rc = run_of[(arch, "shortcut", seed)], run_of[(arch, "clean", seed)]
            def Db(q, sc, rs=rs, rc=rc):
                return Sb[rs][q][sc] - Sb[rc][q][sc]

            def Dp(q, sc, arch=arch, seed=seed):
                m = (pc.arch == arch) & (pc.seed == seed) & (pc.metric == q) & (pc.scope == sc)
                return pc.loc[m, "D"].iloc[0]
            for sc in ["all", "aligned", "conflicting"]:
                lo, hi = pct(Db("mif", sc) - Db("srg", sc))
                bs.append(dict(quantity="A", arch=arch, seed=seed, metric="mif-srg", scope=sc,
                               estimate=Dp("mif", sc) - Dp("srg", sc), ci_lo=lo, ci_hi=hi))
            for q in QS:
                I = Dp(q, "conflicting") - Dp(q, "aligned")
                lo, hi = pct(Db(q, "conflicting") - Db(q, "aligned"))
                bs.append(dict(quantity="I", arch=arch, seed=seed, metric=q, scope="conflicting-aligned",
                               estimate=I, ci_lo=lo, ci_hi=hi))
                gc.append(dict(arch=arch, seed=seed, metric=q, **{f"D_{sc}": Dp(q, sc) for sc in SCOPES}, I=I))
    bsd = pd.DataFrame(bs)
    bsd["resample"] = f"group-stratified image bootstrap (4×200, replace), R={args.n_boot}, seed={args.seed}"
    bsd.to_csv(out / "bootstrap_summary.csv", index=False)
    pd.DataFrame(gc).to_csv(out / "group_comparison.csv", index=False)

    # ---------------- 9-4 견고성
    rb = []
    conds = [("all_baselines", None), ("drop_black", [1, 2, 3, 4])] + \
            [(f"drop_{b}", [i for i in range(5) if i != bi]) for bi, b in enumerate(BASES)]
    for arch in archs:
        for seed in range(3):
            rs, rc = run_of[(arch, "shortcut", seed)], run_of[(arch, "clean", seed)]
            both = (data[rs]["pred"] == data[rs]["y"]) & (data[rc]["pred"] == data[rc]["y"])
            ng = [int(((data[rs]["g"] == k) & both).sum()) for k in range(4)]
            for q in QS:
                for name, bi in conds:
                    Ss, _, nvs, npair = S_point(rs, q, "all", bi)
                    Sc, _, nvc, _ = S_point(rc, q, "all", bi)
                    rb.append(dict(arch=arch, seed=seed, metric=q, condition=name, subset="all800",
                                   n_pairs=npair, S_clean=Sc, S_shortcut=Ss, D=Ss - Sc))
                Ss = S_point(rs, q, "all", None, both)[0]
                Sc = S_point(rc, q, "all", None, both)[0]
                rb.append(dict(arch=arch, seed=seed, metric=q, condition="all_baselines", subset="both_correct",
                               n_pairs=10, S_clean=Sc, S_shortcut=Ss, D=Ss - Sc,
                               **{f"n_{GNAME[k]}": ng[k] for k in range(4)}))
    pd.DataFrame(rb).to_csv(out / "robustness.csv", index=False)

    make_figs(out / "figs", pc, sens, meta, runs, data, archs)
    print_summary(pc, bsd, pd.DataFrame(rb), sens, fa)


# ----------------------------------------------------------------------------- figures
def make_figs(fdir, pc, sens, meta, runs, data, archs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"clean": "#2a78d6", "shortcut": "#eb6834"}
    AC = {"resnet18": "#2a78d6", "resnet50": "#1baf7a"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.edgecolor": "#e4e3df", "font.size": 9, "legend.frameon": False})
    AN = {"resnet18": "ResNet-18", "resnet50": "ResNet-50"}
    # paired plot
    fig, axes = plt.subplots(1, 2 * len(archs), figsize=(3.3 * 2 * len(archs), 3.6), sharey=True)
    for ax, (arch, q) in zip(axes, [(a, q) for q in ["mif", "srg"] for a in archs]):
        d = pc.query("arch==@arch and metric==@q and scope=='all'")
        for _, row in d.iterrows():
            ax.plot([0, 1], [row.S_clean, row.S_shortcut], color="#8a8984", lw=1.5, zorder=1)
            ax.scatter([0], [row.S_clean], color=C["clean"], s=40, zorder=2)
            ax.scatter([1], [row.S_shortcut], color=C["shortcut"], s=40, zorder=2)
        for yv, grp in d.groupby(d.S_shortcut.round(6)):  # 같은 높이 seed 라벨은 한 줄로
            ax.text(1.06, yv, "seed " + ",".join(str(s) for s in grp.seed), va="center", fontsize=7, color="#52514e")
        ax.set_xticks([0, 1], ["clean", "shortcut"])
        ax.set_xlim(-0.3, 1.4)
        ax.set_title(f"{AN[arch]} · {q.upper()}  (mean D={d.D.mean():+.3f})")
    axes[0].set_ylabel("rank sensitivity S = 1 − mean τ-b")
    fig.suptitle("Stage 8-1: paired shortcut–clean rank sensitivity (scope = all 800 images)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fdir / "fig_paired_sensitivity.png", dpi=140)
    plt.close(fig)
    # SRI scatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, q in zip(axes, ["mif", "srg"]):
        d = sens.query("metric==@q and scope=='all'")
        for _, row in d.iterrows():
            ax.scatter(row.sri, row.S, color=AC[row.arch], marker="o" if row.model == "clean" else "^", s=45)
        ax.set_xlabel("SRI (ariel re-evaluation)")
        ax.set_title(f"{q.upper()} rank sensitivity vs SRI")
    axes[0].set_ylabel("S = 1 − mean τ-b")
    hs = [plt.Line2D([], [], color=AC[a], marker="o", ls="", label=AN[a]) for a in archs] + \
         [plt.Line2D([], [], color="#52514e", marker=m, ls="", label=l) for m, l in [("o", "clean ρ0.5"), ("^", "shortcut ρ0.95")]]
    fig.legend(handles=hs, loc="upper right", ncol=4)
    fig.suptitle("Stage 8-2 (exploratory): SRI vs rank sensitivity, 12 models", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fdir / "fig_sri_vs_sensitivity.png", dpi=140)
    plt.close(fig)
    # Random curves
    fig, axes = plt.subplots(len(archs), 5, figsize=(15, 3.2 * len(archs)), sharex=True, sharey=True, squeeze=False)
    for r_i, arch in enumerate(archs):
        for b_i, b in enumerate(BASES):
            ax = axes[r_i, b_i]
            for model in ["clean", "shortcut"]:
                cs = [data[r]["curves"] for r in runs if meta[r]["arch"] == arch and meta[r]["model"] == model]
                x = cs[0][2]
                ax.plot(x, np.mean([c[0][:, METHODS.index("random"), b_i].mean(0) for c in cs], 0),
                        color=C[model], lw=2, label=model)
            ax.set_title(f"{AN[arch]} · {b}")
            if b_i == 0:
                ax.set_ylabel("target prob (random order)")
            if r_i == len(archs) - 1:
                ax.set_xlabel("fraction removed")
    axes[0, 0].legend()
    fig.suptitle("Stage 9-2: Random-order removal curves (R-OMS control), mean over seeds × 800 images", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fdir / "fig_random_curves.png", dpi=140)
    plt.close(fig)


def print_summary(pc, bsd, rb, sens, fa):
    pd.set_option("display.width", 200)
    print("\n[8-1] 주 분석: D = S_shortcut − S_clean (scope=all)")
    t = pc.query("scope=='all'").groupby(["arch", "metric"]).agg(
        S_clean=("S_clean", "mean"), S_shortcut=("S_shortcut", "mean"), D_mean=("D", "mean"),
        D_pos_seeds=("D", lambda v: f"{(v > 0).sum()}/3"), D_by_seed=("D", lambda v: np.round(v.values, 3).tolist()))
    print(t.round(3).to_string())
    print("\n[8-3] arch 평균 D 의 이미지 bootstrap 95% 구간 (scope=all)")
    print(bsd.query("quantity=='D_arch_mean' and scope=='all'")[["arch", "metric", "estimate", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print("\n[8-4] G = S_MIF − S_SRG (scope=all), A = D_MIF − D_SRG")
    print(bsd.query("quantity=='G' and scope=='all'").groupby(["arch", "model"]).estimate.mean().round(3).to_string())
    print(bsd.query("quantity=='A' and scope=='all'")[["arch", "seed", "estimate", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print("\n[9-1] I = D_conflicting − D_aligned")
    print(bsd.query("quantity=='I'")[["arch", "seed", "metric", "estimate", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print("\n[9-4] 견고성: 조건별 평균 D (seed 평균)")
    print(rb.groupby(["arch", "metric", "subset", "condition"]).D.mean().unstack(["arch", "metric"]).round(3).to_string())


if __name__ == "__main__":
    main()
