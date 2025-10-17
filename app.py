# -*- coding: utf-8 -*-
"""
Günlük Parquet-first pipeline (revize):
- sf_crime_y öncelikli okuma ve yoksa sf_crime.csv'den _y üretme (eksik günleri GEOID×gün için 0 ile tamamla)
- 01..10 adımlarını oluşturur; 11'de hızlı bir stacking benzeri model ile tahmin sütunları ekler
- Tüm birleşmeler id-temelli mümkün olduğunda id ile, değilse GEOID(+date) ile yapılır

Notlar:
- Bus/train/police/poi dosyalarının GEOID bazlı özet/mesafe sütunları içerdiği varsayılmıştır.
- Weather şehir geneli olduğundan tarihe göre join edilir.
- Neighbor etkisi için DATA_DIR/neighbors.csv (GEOID,NEIGHBOR_GEOID) beklenir; yoksa adım graceful degrade.
"""
from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple

import os, re, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import streamlit as st

# =========================
# Kurulum / genel yardımcılar
# =========================
st.set_page_config(page_title="Günlük Suç Pipeline (revize)", layout="wide")

try:
    import pyarrow  # noqa
except Exception:
    st.error("Parquet I/O için 'pyarrow' gerekli. Lütfen 'pip install pyarrow' kurun.")
    st.stop()

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "crime_prediction_data_pre"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))
TODAY = pd.Timestamp.today().normalize().date()

#############################
# I/O yardımcıları
#############################

def read_df(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        return pd.read_parquet(p)
    if p.suffix == ".csv" and p.exists():
        return pd.read_csv(p, low_memory=False)
    # otomatik tercih: parquet > csv
    pq = p.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    cs = p.with_suffix(".csv")
    if cs.exists():
        return pd.read_csv(cs, low_memory=False)
    raise FileNotFoundError(str(p))


def write_df(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.suffix != ".parquet":
        p = p.with_suffix(".parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L)
    )


def add_calendar_cols(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        # "datetime" varsa ordan türet
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
            date_col = "date"
        else:
            return df
    d = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["weekday"] = d.dt.weekday
    # basit sezon: DJF=Winter, MAM=Spring, JJA=Summer, SON=Fall
    def _season(m):
        if m in (12,1,2): return "winter"
        if m in (3,4,5): return "spring"
        if m in (6,7,8): return "summer"
        return "fall"
    df["season"] = d.dt.month.map(_season)
    return df

#############################
# 1) sf_crime_y tercih + forward-fill eksik günler
#############################

def ensure_sf_crime_y() -> Path:
    base_csv = DATA_DIR / "sf_crime.csv"
    y_csv    = DATA_DIR / "sf_crime_y.csv"
    y_parq   = y_csv.with_suffix(".parquet")

    if y_csv.exists() or y_parq.exists():
        st.info("✅ sf_crime_y bulundu; bu dosya üzerinden devam edilecek.")
        return y_csv

    if not (base_csv.exists() or base_csv.with_suffix(".parquet").exists()):
        st.error("sf_crime.csv bulunamadı; ilk tohum dosyası gerekli.")
        st.stop()

    df = read_df(base_csv)
    # normalize
    if "GEOID" in df.columns:
        df["GEOID"] = _norm_geoid(df["GEOID"]) 
    # tarih
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    else:
        st.error("sf_crime.csv içinde date/datetime yok.")
        st.stop()

    # eksik günleri GEOID bazında 0 ile tamamla (bugüne kadar)
    df = _fill_missing_days_geoid(df)
    out = write_df(df, y_csv)
    st.success(f"sf_crime_y oluşturuldu → {out.name}")
    return y_csv


def _fill_missing_days_geoid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GEOID" not in df.columns:
        return df
    df["GEOID"] = _norm_geoid(df["GEOID"]) 
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # olay sayısı kolonunu tahmin et
    count_col = "crime_count" if "crime_count" in df.columns else None
    if count_col is None:
        count_col = "crime_count"
        df[count_col] = 1  # olay bazlı ise her satır 1
    # günlük özet (GEOID×date)
    daily = (df.groupby(["GEOID","date"], as_index=False)[count_col].sum())
    # tüm GEOID'ler için tam tarih aralığı
    start = daily["date"].min()
    end   = max(daily["date"].max(), TODAY)
    idx_date = pd.date_range(start, end, freq="D")
    pieces = []
    for g, gdf in daily.groupby("GEOID"):
        gdf = gdf.set_index("date").reindex(idx_date, fill_value=0).rename_axis("date").reset_index()
        gdf.insert(0, "GEOID", g)
        pieces.append(gdf)
    full = pd.concat(pieces, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"]).dt.date
    return full.rename(columns={count_col:"crime_count"})

#############################
# 2) Y_label hesapla ve sf_crime_01
#############################

def build_sf_crime_01(crime_y_path: Path) -> Path:
    df = read_df(crime_y_path)
    df["GEOID"] = _norm_geoid(df["GEOID"]) 
    df = add_calendar_cols(df, "date")
    # GEOID×day×month×season toplam
    agg = (df.groupby(["GEOID","day","month","season"], as_index=False)["crime_count"].sum())
    # GEOID bazında Y_label (≥2 ise 1)
    y_geo = (agg.groupby("GEOID")["crime_count"].sum().reset_index())
    y_geo["Y_label"] = (y_geo["crime_count"] >= 2).astype(int)
    y_geo = y_geo[["GEOID","Y_label"]]
    out = df.merge(y_geo, on="GEOID", how="left")
    p = write_df(out, DATA_DIR / "sf_crime_01.parquet")
    return p

#############################
# 2.5) Genel join yardımcıları
#############################

def _prefer_y(name: str) -> Optional[Path]:
    base = DATA_DIR / name
    y = base.with_name(base.stem + "_y" + base.suffix)
    if y.exists() or y.with_suffix(".parquet").exists():
        return y
    return base


def safe_merge(left: pd.DataFrame, right_path: Path, on: List[str], how: str = "left") -> pd.DataFrame:
    try:
        r = read_df(right_path)
        # normalizasyonlar
        if "GEOID" in on and "GEOID" in r.columns:
            r["GEOID"] = _norm_geoid(r["GEOID"]) 
        if "date" in on and "date" in r.columns:
            r["date"] = pd.to_datetime(r["date"], errors="coerce").dt.date
        return left.merge(r, on=on, how=how)
    except Exception as e:
        st.warning(f"Birleşme atlandı: {right_path.name} — {e}")
        return left

#############################
# 3) 911 → sf_crime_02, 4) 311 → sf_crime_03
#############################

def build_02_03() -> Tuple[Path, Path]:
    d1 = read_df(DATA_DIR / "sf_crime_01.parquet")
    # 911: öncelik _y
    p911 = _prefer_y("sf_911_last_5_year.csv")
    on_cols_911 = [c for c in ["id"] if c in d1.columns]
    if not on_cols_911:
        on_cols_911 = [c for c in ["GEOID","date"] if c in d1.columns]  # fallback
    d2 = safe_merge(d1, p911, on=on_cols_911, how="left")
    p2 = write_df(d2, DATA_DIR / "sf_crime_02.parquet")

    # 311: öncelik _y
    p311 = _prefer_y("sf_311_last_5_years.csv")
    on_cols_311 = [c for c in ["id"] if c in d2.columns]
    if not on_cols_311:
        on_cols_311 = [c for c in ["GEOID","date"] if c in d2.columns]
    d3 = safe_merge(d2, p311, on=on_cols_311, how="left")
    p3 = write_df(d3, DATA_DIR / "sf_crime_03.parquet")
    return p2, p3

#############################
# 4) Nüfus (demografi) → sf_crime_04
#############################

def build_04_population() -> Path:
    d3 = read_df(DATA_DIR / "sf_crime_03.parquet")
    pop = _prefer_y("sf_population.csv")
    try:
        popdf = read_df(pop)
        # GEOID12→GEOID10 normalize (ilk 10 hane)
        if "GEOID" in popdf.columns:
            popdf["GEOID"] = _norm_geoid(popdf["GEOID"], L=10)
        # sol tarafta da 10 hane türet
        d3["GEOID10"] = _norm_geoid(d3["GEOID"], L=10) if "GEOID" in d3.columns else d3.get("GEOID10")
        d4 = d3.merge(popdf, left_on="GEOID10", right_on="GEOID", how="left", suffixes=("","_pop"))
        # eksik nüfus doldur
        if "population" in d4.columns:
            med = pd.to_numeric(d4["population"], errors="coerce").median(skipna=True)
            d4["population"] = pd.to_numeric(d4["population"], errors="coerce").fillna(med if pd.notna(med) else 0)
    except Exception as e:
        st.warning(f"Nüfus birleşmesi atlandı: {e}")
        d4 = d3
    return write_df(d4, DATA_DIR / "sf_crime_04.parquet")

#############################
# 5..8) Bus → Train → POI → Police/Gov
#############################

def build_05_to_08() -> Tuple[Path, Path, Path, Path]:
    d4 = read_df(DATA_DIR / "sf_crime_04.parquet")
    # Bus
    pbus = _prefer_y("sf_bus_stops_with_geoid.csv")
    d5 = safe_merge(d4, pbus, on=[c for c in ["GEOID"] if c in d4.columns])
    p5 = write_df(d5, DATA_DIR / "sf_crime_05.parquet")

    # Train
    ptrain = _prefer_y("sf_train_stops_with_geoid.csv")
    d6 = safe_merge(d5, ptrain, on=[c for c in ["GEOID"] if c in d5.columns])
    p6 = write_df(d6, DATA_DIR / "sf_crime_06.parquet")

    # POI
    ppoi = _prefer_y("sf_pois_cleaned_with_geoid.csv")
    d7 = safe_merge(d6, ppoi, on=[c for c in ["GEOID"] if c in d6.columns])
    p7 = write_df(d7, DATA_DIR / "sf_crime_07.parquet")

    # Police & Government
    ppol = _prefer_y("sf_police_stations.csv")
    d8 = safe_merge(d7, ppol, on=[c for c in ["GEOID"] if c in d7.columns])
    pgov = _prefer_y("sf_government_buildings.csv")
    d8 = safe_merge(d8, pgov, on=[c for c in ["GEOID"] if c in d8.columns])
    p8 = write_df(d8, DATA_DIR / "sf_crime_08.parquet")
    return p5, p6, p7, p8

#############################
# 9) Weather → sf_crime_09 (date join)
#############################

def build_09_weather() -> Path:
    d8 = read_df(DATA_DIR / "sf_crime_08.parquet")
    d8["date"] = pd.to_datetime(d8["date"], errors="coerce").dt.date
    pwe = _prefer_y("sf_weather_5years.csv")
    try:
        w = read_df(pwe)
        if "date" in w.columns:
            w["date"] = pd.to_datetime(w["date"], errors="coerce").dt.date
        elif "datetime" in w.columns:
            w["date"] = pd.to_datetime(w["datetime"], errors="coerce").dt.date
        cols = [c for c in w.columns if c != "GEOID"]
        d9 = d8.merge(w[cols], on="date", how="left")
    except Exception as e:
        st.warning(f"Hava durumu birleşmesi atlandı: {e}")
        d9 = d8
    return write_df(d9, DATA_DIR / "sf_crime_09.parquet")

#############################
# 10) Komşu (neighbor) pencereleri → sf_crime_10
#############################

def build_10_neighbors() -> Path:
    d9 = read_df(DATA_DIR / "sf_crime_09.parquet")
    if not {"GEOID","date"}.issubset(d9.columns):
        st.warning("Neighbor özellikleri için GEOID ve date gerekiyor.")
        return write_df(d9, DATA_DIR / "sf_crime_10.parquet")
    d9["GEOID"] = _norm_geoid(d9["GEOID"]) 
    d9["date"] = pd.to_datetime(d9["date"], errors="coerce").dt.date

    neighbors_path = DATA_DIR / "neighbors.csv"
    if not neighbors_path.exists():
        st.info("neighbors.csv bulunamadı; komşu pencereleri atlanacak.")
        d10 = d9
        return write_df(d10, DATA_DIR / "sf_crime_10.parquet")

    nbr = pd.read_csv(neighbors_path, dtype={"GEOID":str,"NEIGHBOR_GEOID":str})
    nbr["GEOID"] = _norm_geoid(nbr["GEOID"]) 
    nbr["NEIGHBOR_GEOID"] = _norm_geoid(nbr["NEIGHBOR_GEOID"]) 

    daily = (d9.groupby(["GEOID","date"], as_index=False)["crime_count"].sum())
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    # komşuya yansıt
    dd = nbr.merge(daily.rename(columns={"GEOID":"NEIGHBOR_GEOID"}), on="NEIGHBOR_GEOID", how="left")
    dd["date"] = pd.to_datetime(dd["date"]).dt.floor("D")
    dd = dd.sort_values(["GEOID","date"]).reset_index(drop=True)

    def _rolls(g: pd.DataFrame) -> pd.DataFrame:
        x = g.set_index("date").asfreq("D", fill_value=0)
        out = pd.DataFrame(index=x.index)
        out["neighbor_24h"]  = x["crime_count"].rolling("1D").sum().shift(1)
        out["neighbor_3d"]   = x["crime_count"].rolling("3D").sum().shift(1)
        out["neighbor_7d"]   = x["crime_count"].rolling("7D").sum().shift(1)
        out["neighbor_30d"]  = x["crime_count"].rolling("30D").sum().shift(1)
        out["neighbor_365d"] = x["crime_count"].rolling("365D").sum().shift(1)
        out = out.reset_index()
        out["GEOID"] = g["GEOID"].iloc[0]
        return out

    rolls = (dd.groupby("GEOID", group_keys=False).apply(_rolls))
    rolls["date"] = rolls["date"].dt.date
    keep = ["GEOID","date","neighbor_24h","neighbor_3d","neighbor_7d","neighbor_30d","neighbor_365d"]
    d10 = d9.merge(rolls[keep], on=["GEOID","date"], how="left")
    for c in keep[2:]:
        d10[c] = pd.to_numeric(d10[c], errors="coerce").fillna(0.0)
    return write_df(d10, DATA_DIR / "sf_crime_10.parquet")

#############################
# 11) Hızlı stacking-benzeri model ve kayıt
#############################

def build_11_predictions() -> Optional[Path]:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, mean_absolute_error, brier_score_loss
    except Exception as e:
        st.warning(f"Model bağımlılıkları eksik: {e}")
        return None

    d10 = read_df(DATA_DIR / "sf_crime_10.parquet")
    if "date" in d10.columns:
        d10 = d10.sort_values("date")

    y_occ = (pd.to_numeric(d10.get("crime_count", 0), errors="coerce") > 0).astype(int)
    # sayısal özellikler
    drop_cols = {"crime_count","category","subcategory","date","datetime"}
    X = d10.drop(columns=[c for c in drop_cols if c in d10.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    if len(X) < 100:
        st.info("Model için yeterli satır yok (>=100 önerilir).")
        return None

    tscv = TimeSeriesSplit(n_splits=3)
    tr, te = list(tscv.split(X))[-1]

    base = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X.iloc[tr], y_occ.iloc[tr])
    p_hat = clf.predict_proba(X.iloc[te])[:,1]
    st.write("AUC:", float(roc_auc_score(y_occ.iloc[te], p_hat)))
    st.write("Brier:", float(brier_score_loss(y_occ.iloc[te], p_hat)))

    # pozitifler için kuantil regresyon
    pos_idx = np.where(y_occ.values == 1)[0]
    if len(pos_idx) > 50:
        Xm = X.iloc[pos_idx]
        ym = pd.to_numeric(d10.iloc[pos_idx]["crime_count"], errors="coerce").fillna(1)
        q_models = {}
        for q in [0.1, 0.5, 0.9]:
            qr = LGBMRegressor(objective="quantile", alpha=q, n_estimators=400, learning_rate=0.05, random_state=42)
            # train/test split'i zaman ile eşleştir
            tr_pos = Xm.index.intersection(X.index[tr])
            te_pos = Xm.index.intersection(X.index[te])
            if len(tr_pos) > 20 and len(te_pos) > 20:
                qr.fit(Xm.loc[tr_pos], ym.loc[tr_pos])
                if q == 0.5:
                    med_pred = qr.predict(Xm.loc[te_pos])
                    st.write("MAE(q50):", float(mean_absolute_error(ym.loc[te_pos], med_pred)))
                q_models[q] = qr
        p_all = clf.predict_proba(X)[:,1]
        d10["pred_p_occ"] = p_all
        d10["pred_q10"] = q_models.get(0.1).predict(X) if 0.1 in q_models else np.nan
        d10["pred_q50"] = q_models.get(0.5).predict(X) if 0.5 in q_models else np.nan
        d10["pred_q90"] = q_models.get(0.9).predict(X) if 0.9 in q_models else np.nan
        d10["pred_expected"] = d10["pred_p_occ"] * np.nan_to_num(d10["pred_q50"], nan=0.0)
    else:
        d10["pred_p_occ"] = clf.predict_proba(X)[:,1]
        d10["pred_expected"] = d10["pred_p_occ"] * 1.0

    out = write_df(d10, DATA_DIR / "sf_crime_10_with_preds.parquet")
    return out

#############################
# UI
#############################
st.title("📦 Günlük Pipeline (01→10) + Tahmin (11)")

if st.button("▶ 1) sf_crime_y hazırla"):
    ensure_sf_crime_y()

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ 2) Y_label → sf_crime_01"):
        p = build_sf_crime_01(ensure_sf_crime_y())
        st.success(p.name)
with col2:
    if st.button("▶ 3-4) 911/311 → 02/03"):
        p2, p3 = build_02_03()
        st.success(f"{Path(p2).name}, {Path(p3).name}")

if st.button("▶ 5-8) Bus/Train/POI/PoliceGov → 05..08"):
    p5, p6, p7, p8 = build_05_to_08()
    st.success(
        f"{Path(p5).name}, {Path(p6).name}, {Path(p7).name}, {Path(p8).name}")

if st.button("▶ 9) Weather → 09"):
    p9 = build_09_weather()
    st.success(Path(p9).name)

if st.button("▶ 10) Neighbor pencereleri → 10"):
    p10 = build_10_neighbors()
    st.success(Path(p10).name)

if st.button("▶ 11) Tahminleri yaz (stacking-benzeri)"):
    out = build_11_predictions()
    if out:
        st.success(Path(out).name)
    else:
        st.info("Tahmin adımı atlandı.")

# Hızlı dosya görünümü
st.markdown("### 📂 Üretilen dosyalar (Parquet)")
rows = []
for nm in [
    "sf_crime_y.parquet","sf_crime_01.parquet","sf_crime_02.parquet","sf_crime_03.parquet",
    "sf_crime_04.parquet","sf_crime_05.parquet","sf_crime_06.parquet","sf_crime_07.parquet",
    "sf_crime_08.parquet","sf_crime_09.parquet","sf_crime_10.parquet","sf_crime_10_with_preds.parquet",
]:
    p = DATA_DIR / nm
    rows.append({"file": nm, "exists": p.exists(), "size": (p.stat().st_size if p.exists() else 0)})
try:
    dfv = pd.DataFrame(rows)
    st.dataframe(dfv)
except Exception:
    pass
