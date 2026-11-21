"""단계 6 결과 시각화 (CPU, CSV/npz 만 읽음).

입력: outputs/faithfulness/<run>/{summary.csv, curves.npz}
출력: outputs/faithfulness/figs/
  fig1_srg_by_baseline.png   arch × method 별, baseline 5종의 SRG (clean vs shortcut, 막대 = seed 평균, 점 = seed)
  fig2_curves_<arch>.png     baseline 별 MIF(실선)·LIF(점선) 곡선, method 4종 (clean / shortcut 행)
  fig3_trainset_drop.png     Black -> Train-set 으로 바꿨을 때 SRG 변화량 (clean vs shortcut, seed 점)
  fig4_group_srg.png         aligned / conflicting 그룹별 SRG (baseline 별, clean vs shortcut)

사용: python src/plot_faithfulness.py
"""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FROOT = REPO / "outputs" / "faithfulness"
OUT = FROOT / "figs"

# 검증된 categorical 팔레트 (고정 순서). 색은 entity 를 따라감
MODEL_C = {"clean": "#2a78d6", "shortcut": "#eb6834"}
METHOD_C = {"gradcam": "#2a78d6", "ig": "#eb6834", "rise": "#1baf7a", "random": "#eda100"}
METHOD_N = {"gradcam": "Grad-CAM", "ig": "IG", "rise": "RISE", "random": "Random"}
BASES = ["black", "mean", "blur", "inpaint", "trainset"]
BASE_N = {"black": "Black", "mean": "Mean", "blur": "Blur", "inpaint": "Inpaint", "trainset": "Train-set"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False,
})


def load_summary():
    s = pd.concat([pd.read_csv(f) for f in glob.glob(str(FROOT / "*" / "summary.csv"))])
    s["arch"] = s.run.str.split("_").str[0].map({"resnet18": "ResNet-18", "resnet50": "ResNet-50"})
    s["model"] = np.where(s.run.str.contains("rho095"), "shortcut", "clean")
    s["seed"] = s.run.str[-1].astype(int)
    return s


def bars(ax, xs, df, val, width=0.36):
    """clean / shortcut 그룹 막대 (seed 평균) + seed 점"""
    for k, model in enumerate(["clean", "shortcut"]):
        off = (k - 0.5) * width
        d = df[df.model == model]
        m = [d[d.baseline == b][val].mean() for b in BASES]
        ax.bar(xs + off, m, width - 0.04, color=MODEL_C[model], label=model, zorder=2)
        for i, b in enumerate(BASES):
            pts = d[d.baseline == b][val].values
            ax.scatter(np.full(len(pts), xs[i] + off), pts, s=10, color=INK, zorder=3, linewidths=0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)


def fig1(s):
    a = s[s.subset == "all"]
    methods = ["gradcam", "rise", "ig", "random"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    xs = np.arange(len(BASES))
    for r, arch in enumerate(["ResNet-18", "ResNet-50"]):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            bars(ax, xs, a[(a.arch == arch) & (a.method == m)], "srg")
            ax.set_title(f"{arch} · {METHOD_N[m]}")
            ax.set_xticks(xs, [BASE_N[b] for b in BASES], rotation=30)
            if c == 0:
                ax.set_ylabel("SRG (LIF − MIF), higher = better")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.suptitle("SRG by masking baseline — clean (ρ=0.5) vs shortcut (ρ=0.95); bar = mean of 3 seeds, dot = seed", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_srg_by_baseline.png", dpi=140)
    plt.close(fig)


def fig2(arch_key):
    runs = sorted(glob.glob(str(FROOT / f"{arch_key}_*" / "curves.npz")))
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.8), sharex=True, sharey=True)
    for r, model in enumerate(["clean", "shortcut"]):
        tag = "rho050" if model == "clean" else "rho095"
        zs = [np.load(f) for f in runs if tag in f]
        methods = list(zs[0]["methods"])
        bases = list(zs[0]["baselines"])
        x = zs[0]["x"]
        mif = np.mean([z["mif"].mean(0) for z in zs], 0)  # (method, baseline, point)
        lif = np.mean([z["lif"].mean(0) for z in zs], 0)
        for c, b in enumerate(BASES):
            ax = axes[r, c]
            bi = bases.index(b)
            for m in ["gradcam", "rise", "ig", "random"]:
                mi = methods.index(m)
                ax.plot(x, mif[mi, bi], color=METHOD_C[m], lw=2, label=METHOD_N[m] if (r, c) == (0, 0) else None)
                ax.plot(x, lif[mi, bi], color=METHOD_C[m], lw=2, ls=(0, (3, 2)))
            ax.set_title(f"{model} · {BASE_N[b]}")
            if r == 1:
                ax.set_xlabel("fraction of patches removed")
            if c == 0:
                ax.set_ylabel("target-class probability")
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK2, lw=2), plt.Line2D([], [], color=INK2, lw=2, ls=(0, (3, 2)))]
    l += ["MIF (most important first)", "LIF (least important first)"]
    fig.legend(h, l, loc="upper right", ncol=6)
    name = {"resnet18": "ResNet-18", "resnet50": "ResNet-50"}[arch_key]
    fig.suptitle(f"{name}: removal curves (mean over 3 seeds × 800 images)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"fig2_curves_{arch_key}.png", dpi=140)
    plt.close(fig)


def fig3(s):
    a = s[s.subset == "all"].pivot_table(index=["arch", "model", "seed", "method"], columns="baseline",
                                         values="srg").reset_index()
    a["drop"] = a["trainset"] - a["black"]
    methods = ["gradcam", "rise", "ig"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, arch in zip(axes, ["ResNet-18", "ResNet-50"]):
        for i, m in enumerate(methods):
            for k, model in enumerate(["clean", "shortcut"]):
                d = a[(a.arch == arch) & (a.method == m) & (a.model == model)]["drop"]
                xpos = i + (k - 0.5) * 0.3
                ax.bar(xpos, d.mean(), 0.26, color=MODEL_C[model], label=model if i == 0 else None, zorder=2)
                ax.scatter(np.full(len(d), xpos), d, s=12, color=INK, zorder=3, linewidths=0)
                lo = min(d.min(), 0) - 0.008  # 막대·점 아래에 라벨
                ax.text(xpos, lo, f"{d.mean():+.2f}", ha="center", va="top", fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(methods)), [METHOD_N[m] for m in methods])
        ax.set_title(arch)
    axes[0].set_ylabel("ΔSRG (Train-set − Black)")
    axes[0].legend(loc="lower right")
    fig.suptitle("SRG change when switching Black → Train-set baseline (more negative = larger drop)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_trainset_drop.png", dpi=140)
    plt.close(fig)


def fig4(s):
    g = s[s.subset.isin(["aligned", "conflicting"]) & s.method.isin(["gradcam", "rise"])]
    g = g.groupby(["arch", "model", "seed", "subset", "baseline"]).srg.mean().reset_index()  # Grad-CAM·RISE 평균
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    xs = np.arange(len(BASES))
    for r, arch in enumerate(["ResNet-18", "ResNet-50"]):
        for c, sub in enumerate(["aligned", "conflicting"]):
            ax = axes[r, c]
            bars(ax, xs, g[(g.arch == arch) & (g.subset == sub)], "srg")
            ax.set_title(f"{arch} · {sub} ({'background matches class' if sub == 'aligned' else 'background ≠ class'})")
            ax.set_xticks(xs, [BASE_N[b] for b in BASES])
            if c == 0:
                ax.set_ylabel("SRG (mean of Grad-CAM, RISE)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("SRG by group — aligned vs conflicting background", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_group_srg.png", dpi=140)
    plt.close(fig)


def set_korean_font():
    from matplotlib import font_manager
    for name in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Malgun Gothic"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return name
    return None


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("font:", set_korean_font())
    plt.rcParams["axes.unicode_minus"] = False
    s = load_summary()
    fig1(s)
    fig2("resnet18")
    fig2("resnet50")
    fig3(s)
    fig4(s)
    print("saved ->", OUT)
