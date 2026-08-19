"""
@File - global_xai.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 15/08/2026
"""

from __future__ import annotations
from pathlib import Path
import json
from ml.common.preprocess import make_preprocessor
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_PATH = "data/gold/mart_overall_analysis/mart_overall_analysis.parquet"
SCHEMA_PATH = "model/model_schema.json"
OUT_DIR = Path("reports/global_shap_xgb")
TARGET = "o_a_score"
ENV_COL = "env_type"

# Tuned XGB params (from Exp 3, or your best-known). Edit to your values.
BEST_PARAMS = {
    "n_estimators": 1246,
    "learning_rate": 0.05579378978522556,
    "max_depth": 4,
    "min_child_weight": 16.17686906786567,
    "subsample": 0.7266044212478868,
    "colsample_bytree": 0.715268716314553,
    "reg_lambda": 14.155955356878192,
    "reg_alpha": 6.324081256438533,
    "gamma": 0.16930231325154688,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}

THRESHOLD = 1.0

NICE = {
    "appearance": "appearance",
    "tubersize": "tuber size",
    "eveness": "evenness",
    "tubnumbers": "tuber count",
    "eyedepth": "eye depth",
    "yield_": "yield",
    "yield": "yield",
    "uniformity": "uniformity",
    "ffscab": "scab",
    "ffdefects": "defects",
    "ffhollowh": "hollow heart",
    "ff_irs": "rust spot",
    "env_type": "environment",
    "soil_type": "soil type",
    "trial_type": "trial type",
}


def _nice(f):
    return NICE.get(f, f.replace("_", " "))


def _map_encoded_to_original(encoded_names, original_features):
    out = []
    for enc in encoded_names:
        name = enc.split("__", 1)[-1]
        matched = name
        for orig in original_features:
            if name == orig or name.startswith(orig + "_"):
                matched = orig
                break
        out.append(matched)
    return out


def _importance_by_original(sv_subset, orig_map):
    s = (
        pd.DataFrame({"orig": orig_map, "imp": np.abs(sv_subset).mean(axis=0)})
        .groupby("orig")["imp"]
        .sum()
        .sort_values(ascending=False)
    )
    s.index = [_nice(f) for f in s.index]
    return s


def _bar(series, title, path, colour):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(series.index[::-1], series.values[::-1], color=colour)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    for i, v in enumerate(series.values[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = json.load(open(SCHEMA_PATH))
    numeric_cols = schema["features"]["numeric"]
    categorical_cols = schema["features"]["categorical"]
    feature_cols = schema["features"]["feature_cols_order"]

    df = pd.read_parquet(DATA_PATH)
    for c in numeric_cols + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols:
        df[c] = df[c].astype("string")

    miss_frac = df[numeric_cols].isna().mean(axis=1)
    train_mask = df[TARGET].notna() & (miss_frac <= THRESHOLD)
    df_train = df.loc[train_mask].reset_index(drop=True)

    X = df_train[feature_cols]
    y = df_train[TARGET].to_numpy(float)
    env = df_train[ENV_COL].astype("string").str.strip().str.upper().to_numpy()

    pre = make_preprocessor(
        profile="booster", numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )

    Xt = pre.fit_transform(X)
    enc_names = list(pre.get_feature_names_out())
    orig_map = _map_encoded_to_original(enc_names, feature_cols)

    print(f"[shap_xgb] fitting XGB on {len(X):,} rows with tuned params ...")
    model = XGBRegressor(**BEST_PARAMS)
    model.fit(Xt, y)

    print("[shap_xgb] TreeExplainer (exact, all rows) ...")
    explainer = shap.TreeExplainer(model)
    sv = np.array(explainer.shap_values(Xt))
    base = float(explainer.expected_value)
    print(f"[shap_xgb] shap shape {sv.shape}")

    overall = _importance_by_original(sv, orig_map)
    overall.to_csv(OUT_DIR / "importance_overall.csv", header=["mean_abs_shap"])
    _bar(
        overall,
        "Global feature importance — all data (XGBoost)",
        OUT_DIR / "shap_bar_overall.png",
        "#059669",
    )
    print("\nOVERALL:\n", overall.round(4).to_string())

    ne_mask, med_mask = env == "NE", env == "MED"
    ne = _importance_by_original(sv[ne_mask], orig_map)
    med = _importance_by_original(sv[med_mask], orig_map)
    ne.to_csv(OUT_DIR / "importance_ne.csv", header=["mean_abs_shap"])
    med.to_csv(OUT_DIR / "importance_med.csv", header=["mean_abs_shap"])
    _bar(
        ne,
        f"Feature importance — NE (n={ne_mask.sum():,})",
        OUT_DIR / "shap_bar_ne.png",
        "#2563eb",
    )
    _bar(
        med,
        f"Feature importance — MED (n={med_mask.sum():,})",
        OUT_DIR / "shap_bar_med.png",
        "#d97706",
    )
    print(f"\nNE top: {ne.index[0]}   MED top: {med.index[0]}")

    comp = pd.DataFrame({"NE": ne, "MED": med}).fillna(0.0).loc[overall.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    yv = np.arange(len(comp))
    b1 = ax.barh(yv - 0.2, comp["NE"], 0.4, label="NE", color="#2563eb")
    b2 = ax.barh(yv + 0.2, comp["MED"], 0.4, label="MED", color="#d97706")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.set_yticks(yv)
    ax.set_yticklabels(comp.index)
    ax.invert_yaxis()
    ax.set_xlim(0, comp.values.max() * 1.15)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Feature importance by environment — NE vs MED")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "shap_bar_ne_vs_med.png", dpi=150)
    plt.close(fig)
    comp.to_csv(OUT_DIR / "importance_ne_vs_med.csv")

    clean = [
        _nice(o) if o == n.split("__", 1)[-1] else n.split("__", 1)[-1]
        for o, n in zip(orig_map, enc_names)
    ]
    expl_obj = shap.Explanation(
        values=sv,
        base_values=np.repeat(base, len(X)),
        data=np.asarray(Xt),
        feature_names=clean,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(expl_obj, max_display=12, show=False)
    plt.title("SHAP summary (beeswarm) — all data")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_beeswarm_overall.png", dpi=150, bbox_inches="tight")
    plt.close()

    rank_corr = ne.rank(ascending=False).corr(
        med.rank(ascending=False), method="spearman"
    )
    json.dump(
        {
            "base_value": base,
            "n_rows": int(len(X)),
            "n_ne": int(ne_mask.sum()),
            "n_med": int(med_mask.sum()),
            "overall_top": overall.index[0],
            "ne_top": ne.index[0],
            "med_top": med.index[0],
            "ne_med_rank_agreement_spearman": float(rank_corr),
            "params": BEST_PARAMS,
        },
        open(OUT_DIR / "meta.json", "w"),
        indent=2,
    )
    print(f"\nNE vs MED driver-rank agreement (Spearman): {rank_corr:.3f}")
    print(f"Saved to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
