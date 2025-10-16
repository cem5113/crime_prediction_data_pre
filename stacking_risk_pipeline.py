#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stacking-based crime risk pipeline (2-phase, progress, robust exports)

Aşama 1 (TRAIN_PHASE=select):
  - Son 12 ay alt-küme (SUBSET_STRATEGY=last12m, SUBSET_MIN_POS ile güvenlik)
  - TimeSeriesSplit(n_splits=3)
  - Hafif ağaç sayıları (RF/ET/XGB/LGB)
  - LinearSVC/KNN yok
  - Base & Meta için % ilerleme logları
Aşama 2 (TRAIN_PHASE=final):
  - 5 yıl tam veri (SUBSET_STRATEGY=none)
  - TimeSeriesSplit(n_splits=5)
  - Daha yüksek ağaç sayıları
"""

import os, re, json, warnings
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    log_loss, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from joblib import dump, Memory

# -------------------- ENV / Global --------------------

def _env_flag(name: str, default: bool = False) -> int:
    """ENV bayrağını güvenli oku: 1/true/yes/on → 1, aksi → 0 (int döner)."""
    v = os.getenv(name)
    if v is None:
        return 1 if default else 0
    return 1 if str(v).strip().lower() in {"1", "true", "yes", "y", "on"} else 0


CRIME_DIR   = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
GEOID_LEN   = int(os.getenv("GEOID_LEN", "11"))
TRAIN_PHASE = os.getenv("TRAIN_PHASE", "select").strip().lower()  # select | final
CV_JOBS     = int(os.getenv("CV_JOBS", "4"))
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Spatial-TE kontrolü ---
NEIGHBOR_FILE     = os.getenv("NEIGHBOR_FILE", "").strip()        # 'GEOID,neighbor' iki sütunlu csv (opsiyonel)
TE_ALPHA          = float(os.getenv("TE_ALPHA", "50"))            # Laplace smoothing gücü (m)
GEO_COL_NAME      = os.getenv("GEO_COL_NAME", "GEOID")            # GEOID kolon adı

# Eğer kullanıcı ENABLE_SPATIAL_TE vermediyse ama NEIGHBOR_FILE verdiyse,
# Spatial-TE'yi otomatik aç (kullanışlı varsayılan).
_env_has_te       = os.getenv("ENABLE_SPATIAL_TE")
ENABLE_SPATIAL_TE = _env_flag("ENABLE_SPATIAL_TE", default=False)
if _env_has_te is None and NEIGHBOR_FILE:
    ENABLE_SPATIAL_TE = 1

# --- Ablation ayarları ---
ENABLE_TE_ABLATION = _env_flag("ENABLE_TE_ABLATION", default=False)  # 1 → ikinci varyantı da koştur (OHE↔TE)
ABLASYON_BASIS     = os.getenv("ABLASYON_BASIS", "ohe").strip().lower()  # 'ohe' ya da 'te'
if ABLASYON_BASIS not in {"ohe", "te"}:
    ABLASYON_BASIS = "ohe"


def phase_is_select() -> bool:
    return TRAIN_PHASE == "select"

# -------------------- Helpers: date/hour --------------------
def ensure_date_hour_on_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # date
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["date"] = pd.NaT

    # hour_range normalize / impute
    def hr_from_event_hour(s):
        h = pd.to_numeric(s, errors="coerce").fillna(0).astype(int) % 24
        start = (h // 3) * 3
        end = (start + 3) % 24
        return start.map(lambda x: f"{x:02d}") + "-" + end.map(lambda x: f"{x:02d}")

    if "hour_range" in out.columns:
        hr = out["hour_range"].astype(str).str.extract(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
        ok = hr.notna().all(axis=1)
        out.loc[ok, "hour_range"] = (
            hr[0].astype(float).astype(int).map("{:02d}".format) + "-" +
            hr[1].astype(float).astype(int).map("{:02d}".format)
        )
        # kalan NaN'ler için event_hour kullan
        miss = out["hour_range"].isna()
        if miss.any() and "event_hour" in out.columns:
            out.loc[miss, "hour_range"] = hr_from_event_hour(out.loc[miss, "event_hour"])
    elif "event_hour" in out.columns:
        out["hour_range"] = hr_from_event_hour(out["event_hour"])
    else:
        out["hour_range"] = np.nan

    return out

def _load_neighbors(path: str):
    """
    Komşuluk dosyası opsiyonel. Format örnekleri:
      A) GEOID,neighbor
      B) src,dst
    Aynı GEOID için birden çok komşu satırı olabilir.
    Dönen: dict[str, set[str]]
    """
    if not path or not os.path.exists(path):
        return None
    df_n = pd.read_csv(path, dtype=str)
    df_n.columns = [c.lower() for c in df_n.columns]
    if {"geoid", "neighbor"}.issubset(df_n.columns):
        src, dst = "geoid", "neighbor"
    elif {"src", "dst"}.issubset(df_n.columns):
        src, dst = "src", "dst"
    else:
        raise ValueError("NEIGHBOR_FILE beklenen sütunları içermiyor (geoid,neighbor) veya (src,dst)")

    adj = {}
    for g, d in df_n[[src, dst]].itertuples(index=False, name=None):
        if g not in adj:
            adj[g] = set()
        adj[g].add(d)
    return adj

class SpatialTargetEncoder(BaseEstimator, TransformerMixin):
    """
    GEOID için hedef kodlama (komşu katkılı, Laplace smoothing):
      TE(g) = (sum_y(g) + m * global_mean + sum_y(neighbors(g))) / (n(g) + m + n_neighbors(g))
    - m = TE_ALPHA (Laplace smoothing)
    - neighbors(g) varsa eklenir, yoksa yalnız Laplace yapılır.
    Not: CV sırasında sızıntı yok; Pipeline içindeki her fold'un train'inde fit edilir.
    """
    def __init__(self, geo_col=GEO_COL_NAME, alpha=50.0, neighbors_dict=None):
        self.geo_col = geo_col
        self.alpha = float(alpha)
        self.neighbors_dict = neighbors_dict
        self.mapping_ = None
        self.global_mean_ = None

    def _geo_series(self, X):
        # Hem DataFrame hem ndarray destekle
        if isinstance(X, pd.DataFrame):
            if self.geo_col in X.columns:
                return X[self.geo_col].astype(str)
            # ColumnTransformer tek bir sütun seçince isim düşebilir → ilk sütunu al
            if X.shape[1] == 1:
                return X.iloc[:, 0].astype(str)
            raise ValueError(f"{self.geo_col} kolonu bulunamadı ve çoklu sütun geldi.")
        # ndarray yolu
        X = np.asarray(X)
        if X.ndim == 1:
            return pd.Series(X.ravel().astype(str))
        return pd.Series(X[:, 0].astype(str))

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SpatialTargetEncoder.fit için y gerekli.")
        s_geo = self._geo_series(X)
        y = pd.Series(y).astype(float)
        if len(s_geo) != len(y):
            raise ValueError("X ve y uzunlukları uyumsuz.")

        # Grup istatistikleri
        grp = pd.DataFrame({"geo": s_geo, "y": y}).groupby("geo")["y"].agg(["sum", "count"])
        grp.columns = ["sum_y", "n"]
        self.global_mean_ = float(y.mean()) if len(y) else 0.5

        # Komşuluk katkısı
        if self.neighbors_dict:
            sum_dict = grp["sum_y"].to_dict()
            n_dict   = grp["n"].to_dict()
            neigh_sum = []
            neigh_n   = []
            for g in grp.index:
                acc_s = 0.0
                acc_n = 0.0
                for nb in self.neighbors_dict.get(g, []):
                    acc_s += sum_dict.get(nb, 0.0)
                    acc_n += n_dict.get(nb, 0.0)
                neigh_sum.append(acc_s)
                neigh_n.append(acc_n)
            grp["neigh_sum"] = neigh_sum
            grp["neigh_n"]   = neigh_n
        else:
            grp["neigh_sum"] = 0.0
            grp["neigh_n"]   = 0.0

        m = self.alpha
        grp["te"] = (grp["sum_y"] + m*self.global_mean_ + grp["neigh_sum"]) / (grp["n"] + m + grp["neigh_n"])
        self.mapping_ = grp["te"].to_dict()
        return self

    def transform(self, X):
        s_geo = self._geo_series(X)
        te = s_geo.map(self.mapping_).fillna(self.global_mean_ if self.global_mean_ is not None else 0.5)
        return te.to_numpy().reshape(-1, 1)

def subset_last12m(df: pd.DataFrame, min_pos: int = 10_000) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    dmax = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(dmax):
        return df
    sub = df[pd.to_datetime(df["date"], errors="coerce") >= (dmax - pd.Timedelta(days=365))]
    if sub["Y_label"].sum() < min_pos:
        # genişlet: 18 → 24 ay
        for months in (18, 24):
            sub2 = df[pd.to_datetime(df["date"], errors="coerce") >= (dmax - pd.Timedelta(days=30*months))]
            if sub2["Y_label"].sum() >= min_pos:
                sub = sub2
                break
    return sub

# -------------------- File helpers --------------------
def _normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)")[0]
    return s.str[-target_len:].str.zfill(target_len)

def _ensure_date_and_hour_legacy(df: pd.DataFrame) -> pd.DataFrame:
    # Legacy inline converter for sf_crime_08 -> 09
    out = df.copy()
    if "GEOID" in out.columns:
        out["GEOID"] = _normalize_geoid(out["GEOID"], GEOID_LEN)
    out = ensure_date_hour_on_df(out)
    # Impute numerics & cats softly
    count_cols = [c for c in out.columns if any(k in c.lower() for k in ["_count", "911_", "311_"])]
    for c in count_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    num_cols = [c for c in out.columns if out[c].dtype.kind in "fc" and c not in count_cols and c != "Y_label"]
    for c in num_cols:
        med = pd.to_numeric(out[c], errors="coerce").median()
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(med)
    cat_cols = [c for c in out.columns if out[c].dtype == "object"]
    for c in cat_cols:
        mode = out[c].mode(dropna=True)
        out[c] = out[c].fillna(mode.iloc[0] if not mode.empty else "")
    return out

def _make_sf_crime_09_inline():
    src = os.path.join(CRIME_DIR, "sf_crime_08.csv")
    dst = os.path.join(CRIME_DIR, "sf_crime_09.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} bulunamadı (08 yok).")
    df = pd.read_csv(src, low_memory=False, dtype={"GEOID": str})
    df = _ensure_date_and_hour_legacy(df)
    wanted_last = ["date", "hour_range", "GEOID"]
    cols = [c for c in df.columns if c not in wanted_last] + [c for c in wanted_last if c in df.columns]
    df = df[cols]
    Path(CRIME_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"✅ sf_crime_09 hazır → {dst}")
    return dst

def ensure_sf_crime_09() -> str:
    p09 = os.path.join(CRIME_DIR, "sf_crime_09.csv")
    if os.path.exists(p09):
        return p09
    return _make_sf_crime_09_inline()

# -------------------- Feature prep --------------------
def build_feature_lists(df: pd.DataFrame):
    drop_cols = {"Y_label", "id", "datetime", "time"}
    count_like = [c for c in df.columns if any(k in c.lower() for k in ["_count", "911_", "311_"])]
    num_cands  = [c for c in df.columns if df[c].dtype.kind in "fc" and c not in count_like and c not in drop_cols]
    cat_cands  = [c for c in df.columns if df[c].dtype == "object" and c not in drop_cols]
    return count_like, num_cands, cat_cands

def find_leaky_numeric_features(df, feature_cols, y, corr_thr=0.995, auc_thr=0.995):
    X = df[feature_cols].copy()
    leaky, details = set(), {}
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return [], {}
    cmat = pd.concat([X[num_cols], y.rename("Y")], axis=1).corr(numeric_only=True)["Y"].drop(labels=["Y"], errors="ignore")
    for c, v in cmat.abs().items():
        if pd.notna(v) and v >= corr_thr:
            leaky.add(c); details[c] = {"reason": "leak_corr", "corr": float(v), "auc": None}
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        idx = s.notna()
        if idx.sum() < 30:
            continue
        try:
            auc = roc_auc_score(y[idx], s[idx])
            if auc >= auc_thr or auc <= (1 - auc_thr):
                d = details.get(c, {"reason": "leak_auc", "corr": None, "auc": None})
                d["reason"] = "leak_auc"; d["auc"] = float(auc); details[c] = d; leaky.add(c)
        except Exception:
            pass
    return sorted(leaky), details

def build_preprocessor(count_features, num_features, cat_features) -> ColumnTransformer:
    # GEOID varsa ayır: GEOID dışındaki kategoriler OHE, GEOID Spatial-TE (ENABLE_SPATIAL_TE=1 ise)
    cat_features = list(cat_features)  # kopya
    has_geo = GEO_COL_NAME in cat_features

    numeric_pipe_counts = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ])
    numeric_pipe_cont = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    # OHE: nadir kategori birleştirme destekli (mümkünse)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe_other = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    transformers = [
        ("cnt", numeric_pipe_counts, count_features),
        ("num", numeric_pipe_cont,   num_features),
    ]

    neighbors_dict = _load_neighbors(NEIGHBOR_FILE) if (ENABLE_SPATIAL_TE and has_geo) else None

    if ENABLE_SPATIAL_TE and has_geo:
        # GEOID → Spatial-TE
        transformers.append(
            ("geo_te", SpatialTargetEncoder(geo_col=GEO_COL_NAME, alpha=TE_ALPHA, neighbors_dict=neighbors_dict), [GEO_COL_NAME])
        )
        # Diğer kategoriler OHE
        other_cats = [c for c in cat_features if c != GEO_COL_NAME]
        if other_cats:
            transformers.append(("cat", categorical_pipe_other, other_cats))
    else:
        # Hepsi OHE
        if cat_features:
            transformers.append(("cat", categorical_pipe_other, cat_features))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    return pre

def build_preprocessor_forced(count_features, num_features, cat_features, use_spatial_te: bool) -> ColumnTransformer:
    """
    Mevcut global ENABLE_SPATIAL_TE değerini bozmadan, geçici olarak
    (use_spatial_te ? 1 : 0) ile preprocessor üret.
    """
    global ENABLE_SPATIAL_TE
    prev = ENABLE_SPATIAL_TE
    try:
        ENABLE_SPATIAL_TE = 1 if use_spatial_te else 0
        return build_preprocessor(count_features, num_features, cat_features)
    finally:
        ENABLE_SPATIAL_TE = prev

# -------------------- Models --------------------
def base_estimators(class_weight_balanced=True):
    cw = "balanced" if class_weight_balanced else None
    light = phase_is_select()

    ests = []
    ests.append(("lr_l1", LogisticRegression(
        penalty="l1", solver="saga", C=0.5, max_iter=3000, class_weight=cw)))
    ests.append(("ridge", RidgeClassifier(alpha=1.0, class_weight=cw)))

    ests.append(("rf", RandomForestClassifier(
        n_estimators=150 if light else 400, max_depth=14, min_samples_leaf=3,
        class_weight="balanced_subsample" if class_weight_balanced else None,
        n_jobs=-1, random_state=42)))
    ests.append(("et", ExtraTreesClassifier(
        n_estimators=200 if light else 500, max_depth=14, min_samples_leaf=3,
        class_weight=cw, n_jobs=-1, random_state=42)))
    ests.append(("hgb", HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.07 if light else 0.06, max_bins=255,
        l2_regularization=0.0, random_state=42)))

    # Optional boosters
    HAS_XGB = False
    HAS_LGB = False
    try:
        from xgboost import XGBClassifier  # type: ignore
        HAS_XGB = True
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        HAS_LGB = True
    except Exception:
        pass

    if HAS_XGB:
        from xgboost import XGBClassifier
        ests.append(("xgb", XGBClassifier(
            n_estimators=200 if light else 400, max_depth=6, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42)))
    if HAS_LGB:
        from lightgbm import LGBMClassifier
        ests.append(("lgb", LGBMClassifier(
            n_estimators=300 if light else 600, num_leaves=31, learning_rate=0.07 if light else 0.05,
            subsample=0.9, colsample_bytree=0.9, min_data_in_leaf=50,
            force_col_wise=True, verbosity=-1, objective="binary",
            class_weight=cw, n_jobs=-1, random_state=42)))
    return ests

# -------------------- Metrics & CV with progress --------------------
def metric_row(name, y_true, y_hat, proba):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.c_[1-proba, proba])),
        "brier": float(brier_score_loss(y_true, proba)),
    }

def optimal_threshold(y_true, proba, beta=1.0):
    prec, rec, th = precision_recall_curve(y_true, proba)
    f = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
    if len(th) == 0:
        return 0.5
    return float(th[np.nanargmax(f[:-1])])

def make_cv(df: pd.DataFrame, folds_select: int = 3, folds_final: int = 5):
    n_splits = folds_select if phase_is_select() else folds_final
    if "date" in df.columns and df["date"].notna().any():
        return TimeSeriesSplit(n_splits=n_splits), None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None

def cv_oof_and_metrics(pipes: dict, X: pd.DataFrame, y: pd.Series, cv, cv_jobs: int = 4):
    # Manuel CV ile % ilerleme
    splits = list(cv.split(X, y))
    n_folds = len(splits)
    n_models = len(pipes)
    total = n_folds * n_models
    done = 0

    rows, oof_probs, names = [], {}, []
    for name in pipes.keys():
        oof_probs[name] = np.zeros(len(X), dtype=np.float32)

    for m_i, (name, pipe) in enumerate(pipes.items(), 1):
        names.append(name)
        for f_i, (tr, te) in enumerate(splits, 1):
            mdl = clone(pipe)
            mdl.fit(X.iloc[tr], y.iloc[tr])
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X.iloc[te])[:, 1]
            else:
                d = mdl.decision_function(X.iloc[te])
                p = (d - d.min()) / (d.max() - d.min() + 1e-9)
            oof_probs[name][te] = p
            done += 1
            print(f"Progress: {done}/{total} ({100.0*done/total:5.1f}%) — [{name}] fold {f_i}/{n_folds}")

        # Model bazında OOF metrik
        proba = oof_probs[name]
        thr = optimal_threshold(y, proba)
        yhat = (proba >= thr).astype(int)
        rows.append(metric_row(name, y, yhat, proba))

    Z = np.column_stack([oof_probs[n] for n in names])
    return pd.DataFrame(rows), Z, names, oof_probs

def evaluate_meta_on_oof(Z: np.ndarray, y: pd.Series):
    folds = 3 if phase_is_select() else 5
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = []

    models = [("Stacking(meta=LogReg)", LogisticRegression(max_iter=5000))]
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        models.append(("Stacking(meta=LightGBM)", LGBMClassifier(
            n_estimators=300 if phase_is_select() else 400, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_data_in_leaf=50,
            force_col_wise=True, verbosity=-1, random_state=42
        )))
    except Exception:
        pass

    for label, meta in models:
        ps = np.zeros(len(y), dtype=float)
        for i, (tr, te) in enumerate(cv.split(Z, y), 1):
            meta.fit(Z[tr], y.iloc[tr])
            p = meta.predict_proba(Z[te])[:, 1] if hasattr(meta, "predict_proba") else meta.decision_function(Z[te])
            ps[te] = p
            print(f"Meta progress [{label}]: {i}/{folds} ({100*i/folds:5.1f}%)")

        thr = optimal_threshold(y, ps)
        yhat = (ps >= thr).astype(int)
        m = metric_row(label, y, yhat, ps); m["model"] = label
        results.append(m)

    return pd.DataFrame(results)

# -------------------- Fit full & export --------------------
def fit_full_models_and_export(
    pipes: dict,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    choose_meta: str = "logreg",
):
    proba_cols, mats, full_pipes = [], [], {}
    for name, base in pipes.items():
        base.fit(X, y)
        full_pipes[name] = base
        if hasattr(base, "predict_proba"):
            p = base.predict_proba(X)[:, 1]
        else:
            d = base.decision_function(X)
            p = (d - d.min()) / (d.max() - d.min() + 1e-9)
        mats.append(p.reshape(-1, 1)); proba_cols.append(name)
    Z_full = np.hstack(mats)

    if choose_meta == "lgb":
        try:
            from lightgbm import LGBMClassifier
            meta = LGBMClassifier(
                n_estimators=300 if phase_is_select() else 400, num_leaves=31, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, min_data_in_leaf=50,
                force_col_wise=True, verbosity=-1, random_state=42
            )
            meta_name = "meta_lgb"
        except Exception:
            meta = LogisticRegression(max_iter=5000); meta_name = "meta_logreg"
    else:
        meta = LogisticRegression(max_iter=5000); meta_name = "meta_logreg"

    meta.fit(Z_full, y)

    out_models = Path(CRIME_DIR) / "models"
    out_models.mkdir(parents=True, exist_ok=True)
    dump(preprocessor, out_models / "preprocessor.joblib")
    dump(full_pipes,   out_models / "base_pipes.joblib")
    dump({"names": proba_cols, "meta": meta}, out_models / f"stacking_{meta_name}.joblib")

    p_stack = meta.predict_proba(Z_full)[:, 1] if hasattr(meta, "predict_proba") else meta.decision_function(Z_full)
    thr = optimal_threshold(y, p_stack)
    (out_models / f"threshold_{meta_name}.json").write_text(json.dumps({"threshold": float(thr)}), encoding="utf-8")
    return meta_name, proba_cols, p_stack, thr

# -------------------- Exports -------------------- 
def export_risk_tables(df, y, proba, threshold, out_prefix=""):
    df = ensure_date_hour_on_df(df)

    cols = [c for c in ["GEOID", "date", "hour_range"] if c in df.columns]
    risk = df[cols].copy()
    risk["risk_score"] = proba

    # tarih boşsa bugünü bas (SF yerel tarihi)
    if "date" not in risk.columns or pd.to_datetime(risk["date"], errors="coerce").isna().all():
        from datetime import datetime
        from zoneinfo import ZoneInfo
        risk["date"] = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    
    # hour_range formatını garanti altına al (00-03 gibi)
    if "hour_range" in risk.columns:
        hr = risk["hour_range"].astype(str).str.extract(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
        ok = hr.notna().all(axis=1)
        risk.loc[ok, "hour_range"] = (
            hr[0].astype(float).astype(int).map("{:02d}".format) + "-" +
            hr[1].astype(float).astype(int).map("{:02d}".format)
        )
    
    # tekilleştir (aynı GEOID-date-hour_range tekrarları varsa)
    if set(["GEOID", "date", "hour_range"]).issubset(risk.columns):
        risk = risk.drop_duplicates(["GEOID", "date", "hour_range"], keep="last")

    # seviyeler
    q80 = risk["risk_score"].quantile(0.8)
    q90 = risk["risk_score"].quantile(0.9)
    def level(x):
        if x >= max(threshold, q90):
            return "critical"
        elif x >= max(threshold * 0.8, q80):
            return "high"
        elif x >= 0.5 * threshold:
            return "medium"
        else:
            return "low"

    risk["risk_level"] = risk["risk_score"].apply(level)
    risk["risk_decile"] = pd.qcut(risk["risk_score"].rank(method="first"), 10, labels=False) + 1

    risk_path = os.path.join(CRIME_DIR, f"risk_hourly{out_prefix}.csv")
    risk.to_csv(risk_path, index=False)
    rec_path = None

    # aynı gün için Top-K devriye önerileri
    if "date" in risk.columns and "hour_range" in risk.columns:
        latest_date = pd.to_datetime(risk["date"], errors="coerce").max()
        if pd.notna(latest_date):
            latest_date = latest_date.date()
            day = risk[pd.to_datetime(risk["date"], errors="coerce").dt.date == latest_date].copy()
            rec_rows = []
            TOP_K = int(os.getenv("PATROL_TOP_K", "50"))
            for hr in sorted(day["hour_range"].dropna().unique()):
                slot = day[day["hour_range"] == hr].sort_values("risk_score", ascending=False)
                for _, r in slot.head(TOP_K).iterrows():
                    rec_rows.append({
                        "date": latest_date,
                        "hour_range": hr,
                        "GEOID": r.get("GEOID", ""),
                        "risk_score": float(r["risk_score"]),
                        "risk_level": r["risk_level"],
                    })
            recs = pd.DataFrame(rec_rows)
            rec_path = os.path.join(CRIME_DIR, f"patrol_recs{out_prefix}.csv")
            recs.to_csv(rec_path, index=False)
            print(f"Patrol recs → {rec_path}")
        else:
            print("ℹ️ Patrol önerisi atlandı (geçerli tarih bulunamadı).")
    else:
        print("ℹ️ Patrol önerisi atlandı (date/hour_range yok).")

    return risk_path, rec_path
  
def optional_top_crime_types():
    raw_path = os.path.join(CRIME_DIR, "sf_crime.csv")
    if not os.path.exists(raw_path): return None
    dfc = pd.read_csv(raw_path, low_memory=False, dtype={"GEOID": str})
    if "category" not in dfc.columns: return None
    if "date" in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc["date"], errors="coerce")
    elif "datetime" in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc["datetime"], errors="coerce")
    else:
        return None
    dfc = dfc.dropna(subset=["date"])
    latest = dfc["date"].max(); start = latest - pd.Timedelta(days=30)
    df30 = dfc[(dfc["date"] >= start) & (dfc["date"] <= latest)].copy()
    if "hour_range" not in df30.columns:
        if "event_hour" in df30.columns:
            hr = (pd.to_numeric(df30["event_hour"], errors="coerce").fillna(0).astype(int) // 3) * 3
            df30["hour_range"] = hr.apply(lambda x: f"{int(x):02d}-{int((x+3)%24):02d}")
        else:
            df30["hour_range"] = "00-03"
    grp = (df30.groupby(["GEOID", "hour_range", "category"]).size()
                .reset_index(name="n")).sort_values(["GEOID","hour_range","n"], ascending=[True, True, False])
    top3 = grp.groupby(["GEOID","hour_range"], as_index=False).head(3)
    out = top3.groupby(["GEOID", "hour_range"])["category"].apply(list).reset_index(name="top3_categories")
    path = os.path.join(CRIME_DIR, "risk_types_top3.csv"); out.to_csv(path, index=False); return path

# -------------------- Main --------------------
if __name__ == "__main__":
    dataset_env = os.getenv("STACKING_DATASET", "")
    if dataset_env and os.path.exists(dataset_env):
        data_path = dataset_env
    else:
        data_path = ensure_sf_crime_09()

    print(f"📄 Using dataset: {data_path} | TRAIN_PHASE={TRAIN_PHASE} | CV_JOBS={CV_JOBS}")
    df = pd.read_csv(data_path, low_memory=False, dtype={"GEOID": str})
    if "Y_label" not in df.columns:
        raise ValueError("Y_label kolonu bulunamadı.")

    # Tarih/saat garantile
    df = ensure_date_hour_on_df(df)

    # Seçim aşamasında son 12 ay alt-küme
    if phase_is_select():
        before = df.shape
        min_pos = int(os.getenv("SUBSET_MIN_POS", "10000"))
        strategy = os.getenv("SUBSET_STRATEGY", "last12m").lower()
        if strategy == "last12m":
            df = subset_last12m(df, min_pos=min_pos)
        after = df.shape
        print(f"🔧 Subset ({strategy}): {before} → {after} (pos={int(df['Y_label'].sum())})")

    # Feature lists
    counts, nums, cats = build_feature_lists(df)
    all_feats = list(dict.fromkeys(counts + nums + cats))

    # Leakage avcısı
    sus, info = find_leaky_numeric_features(df, all_feats, df["Y_label"].astype(int))
    if sus:
        print("⚠️ Olası sızıntı/overfit üreten numeric kolonlar DROP ediliyor:")
        for c in sus:
            meta = info.get(c, {})
            corr = f"(corr≈{meta.get('corr'):.4f})" if meta.get("corr") is not None else ""
            aucv = f"(auc≈{meta.get('auc'):.4f})" if meta.get("auc")  is not None else ""
            print(f"  - {c} → reason={meta.get('reason')} {corr} {aucv}")
        counts = [c for c in counts if c not in sus]
        nums   = [c for c in nums   if c not in sus]
        all_feats = list(dict.fromkeys(counts + nums + cats))

    # Preprocessor
    pre = build_preprocessor(counts, nums, cats)

    # Pre + VT + L1 raporu
    try:
        pre2 = clone(pre); pre2.fit(df[all_feats], df["Y_label"].astype(int))
        try:
            pre_names = pre2.get_feature_names_out()
        except Exception:
            pre_names = np.array([f"f{i}" for i in range(pre2.transform(df[all_feats][:1]).shape[1])])
        Xt = pre2.transform(df[all_feats])
        vt = VarianceThreshold(0.0).fit(Xt)
        vt_removed = pre_names[~vt.get_support()].tolist()
        kept_after_vt = pre_names[vt.get_support()]
        l1_removed = []
        try:
            l1 = SelectFromModel(LogisticRegression(penalty="l1", solver="saga", C=0.2, max_iter=4000), max_features=300)
            Xt_vt = vt.transform(Xt); l1.fit(Xt_vt, df["Y_label"].astype(int))
            l1_removed = np.array(kept_after_vt)[~l1.get_support()].tolist()
        except Exception:
            pass
        if vt_removed:
            print(f"🧹 VarianceThreshold ile ELENEN {len(vt_removed)} özellik (transformed)…")
        if l1_removed:
            print(f"🔻 L1 tabanlı seçim ile ELENEN {len(l1_removed)} özellik (transformed)…")
    except Exception:
        pass

    # X/y
    feature_cols = list(dict.fromkeys(counts + nums + cats))
    X = df[feature_cols].copy()
    num_like = [c for c in feature_cols if (c in counts) or (c in nums)]
    if num_like:
        X[num_like] = X[num_like].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    y = df["Y_label"].astype(int)

    # Pipelines
    MEM = Memory(location=os.path.join(CRIME_DIR, ".skcache"), verbose=0)
    base_list = base_estimators(class_weight_balanced=True)
    base_pipes = {name: Pipeline([("prep", pre), ("clf", est)], memory=MEM) for name, est in base_list}
    print(f"🧠 Pipeline cache: {MEM.location}")

    # CV nesnesi
    cv, _ = make_cv(df)

    # Base + OOF + % ilerleme
    print("\n🔎 Evaluating base + OOF (with progress)…")
    base_metrics, Z, base_names, oof_map = cv_oof_and_metrics(base_pipes, X, y, cv, cv_jobs=CV_JOBS)
    Path(CRIME_DIR).mkdir(parents=True, exist_ok=True)
    base_metrics.to_csv(os.path.join(CRIME_DIR, "metrics_base.csv"), index=False)
    np.savez_compressed(os.path.join(CRIME_DIR, "oof_base_probs.npz"), **oof_map)
    print(base_metrics)

    # Meta + % ilerleme
    print("\n🤝 Evaluating stacking meta on OOF…")
    meta_metrics = evaluate_meta_on_oof(Z, y)
    meta_metrics.to_csv(os.path.join(CRIME_DIR, "metrics_stacking.csv"), index=False)
    print(meta_metrics)

    # --- Spatial-TE Ablation (isteğe bağlı) ---
    if ENABLE_TE_ABLATION:
        # İkinci koşunun hangi tabanı olacağı: 'ohe' → OHE bazlı; 'te' → Spatial-TE bazlı
        basis = ABLASYON_BASIS  # 'ohe' | 'te'
        use_spatial_in_alt = (basis == "te")

        print(f"\n🧪 Ablation: ikinci varyant = {basis.upper()} (use_spatial_te={use_spatial_in_alt})")

        # Alternatif preprocessor
        pre_alt = build_preprocessor_forced(counts, nums, cats, use_spatial_te=use_spatial_in_alt)

        # Alternatif pipeline seti
        base_list_alt = base_estimators(class_weight_balanced=True)
        base_pipes_alt = {name: Pipeline([("prep", pre_alt), ("clf", est)], memory=MEM) for name, est in base_list_alt}

        # Aynı CV ile koştur
        print("\n🔎 [Ablation] Evaluating base + OOF (with progress)…")
        base_metrics_alt, Z_alt, base_names_alt, oof_map_alt = cv_oof_and_metrics(base_pipes_alt, X, y, cv, cv_jobs=CV_JOBS)

        # Dosya adlarında suffix
        suf = f"_{basis}"
        base_metrics_alt.to_csv(os.path.join(CRIME_DIR, f"metrics_base{suf}.csv"), index=False)
        np.savez_compressed(os.path.join(CRIME_DIR, f"oof_base_probs{suf}.npz"), **oof_map_alt)
        print(base_metrics_alt)

        print("\n🤝 [Ablation] Evaluating stacking meta on OOF…")
        meta_metrics_alt = evaluate_meta_on_oof(Z_alt, y)
        meta_metrics_alt.to_csv(os.path.join(CRIME_DIR, f"metrics_stacking{suf}.csv"), index=False)
        print(meta_metrics_alt)

        # Kısa özet karşılaştırma dosyası
        best_main = meta_metrics.sort_values("pr_auc", ascending=False).iloc[0]
        best_alt  = meta_metrics_alt.sort_values("pr_auc", ascending=False).iloc[0]
        ablation_df = pd.DataFrame([
            {"variant": "current(TE_on)" if ENABLE_SPATIAL_TE else "current(OHE)",
             "best_model": best_main["model"], "pr_auc": best_main["pr_auc"],
             "roc_auc": best_main["roc_auc"], "brier": best_main["brier"]},
            {"variant": f"ablation({basis})",
             "best_model": best_alt["model"], "pr_auc": best_alt["pr_auc"],
             "roc_auc": best_alt["roc_auc"], "brier": best_alt["brier"]},
        ])
        ablation_df.to_csv(os.path.join(CRIME_DIR, "ablation_spatial_te.csv"), index=False)
        print("🧾 Ablation summary → ablation_spatial_te.csv")
  
    # Meta seçimi
    best_row = meta_metrics.sort_values("pr_auc", ascending=False).iloc[0]
    chosen_meta = "lgb" if ("LightGBM" in best_row["model"]) else "logreg"
    print(f"\n✅ Chosen meta: {chosen_meta} (by PR-AUC)")

    # Full fit & export
    meta_name, proba_cols, p_stack, thr = fit_full_models_and_export(base_pipes, pre, X, y, choose_meta=chosen_meta)
    print(f"Saved models. Threshold ({meta_name}) = {thr:.4f}")

    # Risk & patrol (güvenli)
    risk_path, rec_path = export_risk_tables(df, y, p_stack, thr)
    print(f"Risk table → {risk_path}")
    if rec_path:
        print(f"Patrol recs → {rec_path}")

    # Top crime types (opsiyonel)
    types_path = optional_top_crime_types()
    if types_path: print(f"Top crime types → {types_path}")

    # Birleşik metrik
    try:
        base_metrics["group"] = "base"; meta_metrics["group"] = "stacking"
        allm = pd.concat([base_metrics, meta_metrics], ignore_index=True)
        allm.to_csv(os.path.join(CRIME_DIR, "metrics_all.csv"), index=False)
        print("All metrics → metrics_all.csv")
    except Exception as e:
        print(f"[WARN] metrics merge failed: {e}")
