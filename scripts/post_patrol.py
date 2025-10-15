#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEOID komşuluk dosyası üretir.
- Veri setindeki GEOID uzunluğu (GEOID_LEN) ile uyumlu hale getirir.
- Poligon katmanındaki GEOID alanını otomatik bulur (veya POLY_GEO_COL ile zorla).
- Eşleşmeyi hem 'ilk L' hem 'son L' den dener; en yüksek örtüşmeyi seçer.
- Queen contiguity (en az bir nokta temas) ile komşuluk kurar.
Çıktı: crime_prediction_data/neighbors.csv  (iki sütun: GEOID,neighbor)
"""

import os
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd

CRIME_DIR    = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
DATASET      = os.getenv("STACKING_DATASET", str(Path(CRIME_DIR) / "sf_crime_grid_full_labeled.csv"))
GEOID_LEN    = int(os.getenv("GEOID_LEN", "11"))
OUT_PATH     = str(Path(CRIME_DIR) / "neighbors.csv")
POLY_FILE    = os.getenv("POLY_FILE", "").strip()      # ör: crime_prediction_data/sf_census_blocks_with_population.geojson
POLY_GEO_COL = os.getenv("POLY_GEO_COL", "").strip()   # ör: GEOID

# ---------- yardımcılar ----------
def _only_digits(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D+", "", regex=True)

def _norm_left(s: pd.Series, L: int) -> pd.Series:
    d = _only_digits(s)
    return d.str.slice(0, L).str.zfill(L)

def _norm_right(s: pd.Series, L: int) -> pd.Series:
    d = _only_digits(s)
    return d.str[-L:].str.zfill(L)

def _geo_candidates(cols):
    # GEOID alanı için olası isimler (öncelik sırası)
    pri = [
        "geoid", "geoid20", "geoid10",
        "tract_geoid", "tractgeoid", "tractid", "tract",
        "geo_id", "geoid_t", "geoid_tract",
        "block_geoid", "blockid", "blockce", "blockce10", "blockce20"
    ]
    cl = {c.lower(): c for c in cols}
    for k in pri:
        if k in cl:
            return cl[k]
    for c in cols:
        if re.search(r"geo|tract|block", c, re.I):
            return c
    return None

def load_dataset_keys() -> set[str]:
    if not os.path.exists(DATASET):
        raise SystemExit(f"Veri seti yok: {DATASET}")
    df = pd.read_csv(DATASET, dtype={"GEOID": str}, low_memory=False)
    if "GEOID" not in df.columns:
        raise SystemExit("STACKING_DATASET içinde 'GEOID' kolonu yok.")
    keys = _only_digits(df["GEOID"]).str[-GEOID_LEN:].str.zfill(GEOID_LEN).unique().tolist()
    return set(keys)

def find_polygon_layer(keys: set[str]):
    cands = [Path(POLY_FILE)] if POLY_FILE else []
    if not cands:
        for e in ("*.geojson","*.gpkg","*.shp","*.parquet"):
            cands += list(Path(CRIME_DIR).glob(e))

    best = None
    best_overlap = 0
    best_mode = None  # "left" or "right"
    best_geocol = None

    for p in cands:
        try:
            gdf = gpd.read_file(p)
            geocol = POLY_GEO_COL or _geo_candidates(list(gdf.columns))
            if not geocol:
                continue
            series = gdf[geocol].astype(str)
            left  = set(_norm_left(series,  GEOID_LEN).unique())
            right = set(_norm_right(series, GEOID_LEN).unique())
            ol = len(keys & left)
            or_ = len(keys & right)
            if max(ol, or_) > best_overlap:
                best = p
                best_overlap = max(ol, or_)
                best_mode = "left" if ol >= or_ else "right"
                best_geocol = geocol
        except Exception:
            continue

    if not best or best_overlap == 0:
        raise SystemExit("Polygon katmanı bulunamadı: crime_prediction_data içinde GEOID örtüşmesi yakalanamadı.")
    print(f"✅ Poligon: {best.name} | GEOID sütunu: {best_geocol} | normalize: {best_mode} {GEOID_LEN}")
    return str(best), best_geocol, best_mode

def build_neighbors(poly_path: str, geocol: str, mode: str, keys: set[str]):
    gdf = gpd.read_file(poly_path)[[geocol, "geometry"]].copy()
    gdf["GKEY"] = _norm_left(gdf[geocol], GEOID_LEN) if mode == "left" else _norm_right(gdf[geocol], GEOID_LEN)
    gdf = gdf[gdf["GKEY"].isin(keys)].dropna(subset=["geometry"]).reset_index(drop=True)

    # metrik CRS + spatial index
    try:
        gdf = gdf.to_crs(3857)
    except Exception:
        pass
    sidx = gdf.sindex

    pairs = set()
    for i, geom in enumerate(gdf.geometry):
        for j in sidx.intersection(geom.bounds):
            if j <= i:
                continue
            g2 = gdf.geometry.iloc[j]
            # Queen contiguity: sınır veya köşe teması
            if geom.touches(g2) or geom.boundary.intersects(g2.boundary):
                gi = gdf["GKEY"].iloc[i]; gj = gdf["GKEY"].iloc[j]
                if gi != gj:
                    pairs.add((gi, gj)); pairs.add((gj, gi))

    out = pd.DataFrame(sorted(pairs), columns=["GEOID", "neighbor"])
    Path(CRIME_DIR).mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Neighbors → {OUT_PATH} | edge_count={len(out)}")
    if not out.empty:
        print(out.head(10).to_string(index=False))
    return OUT_PATH

if __name__ == "__main__":
    keys = load_dataset_keys()
    poly_path, geocol, mode = find_polygon_layer(keys)
    build_neighbors(poly_path, geocol, mode, keys)
