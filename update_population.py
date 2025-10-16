# update_population.py — nüfus (GEOID + population) zenginleştirme → sf_crime_03.csv

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ----------------------------- Helpers -----------------------------
def _digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")

def _mode_len(series: pd.Series) -> int:
    if series.empty:
        return 11
    L = series.astype(str).str.len()
    m = L.mode(dropna=True)
    return int(m.iloc[0]) if not m.empty else int(L.dropna().median())

def _key(series: pd.Series, L: int) -> pd.Series:
    s = _digits_only(series).str.replace(" ", "", regex=False)
    return s.str.zfill(L).str[:L]

def _find_geoid_col(df: pd.DataFrame) -> str | None:
    cands = [
        "GEOID","geoid","geo_id","GEOID10","geoid10","GeoID",
        "tract","TRACT","tract_geoid","TRACT_GEOID",
        "geography_id","GEOID2",
    ]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low:
            return low[n.lower()]
    # fallback: içinde 'geoid' geçen herhangi bir kolon
    for c in df.columns:
        if "geoid" in c.lower():
            return c
    return None

def _find_population_col(df: pd.DataFrame) -> str | None:
    cands = [
        "population","pop","total_population","B01003_001E","estimate","total",
    ]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low:
            return low[n.lower()]
    # çok sade dosyalarda 'value' gibi gelebilir
    for c in df.columns:
        if re.fullmatch(r"(pop.*|.*population.*|value)", c, flags=re.I):
            return c
    return None

def _len_ok(s: pd.Series, L: int) -> float:
    s = s.fillna("").astype(str)
    return float((s.str.len() == L).mean())

def _level_name(L: int) -> str:
    return {5: "county", 11: "tract", 12: "blockgroup", 15: "block"}.get(L, f"L={L}")

# ----------------------------- Paths/ENV -----------------------------
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Crime input otomatik bul
CRIME_INPUT = os.getenv("CRIME_INPUT", "") or None
if not CRIME_INPUT:
    for p in [BASE_DIR / "sf_crime_02.csv", Path("sf_crime_02.csv"),
              BASE_DIR / "sf_crime.csv",     Path("sf_crime.csv")]:
        if p.exists():
            CRIME_INPUT = str(p); break
if not CRIME_INPUT or not Path(CRIME_INPUT).exists():
    raise FileNotFoundError("CRIME_INPUT bulunamadı. 'sf_crime_02.csv' veya 'sf_crime.csv' gereklidir.")

CRIME_OUTPUT = str(BASE_DIR / "sf_crime_03.csv")

# Population input: sadece YEREL CSV
POPULATION_PATH = (os.getenv("POPULATION_PATH", "") or "").strip()
if not POPULATION_PATH:
    cand = BASE_DIR / "sf_population.csv"
    if cand.exists():
        POPULATION_PATH = str(cand)
    elif Path("sf_population.csv").exists():
        POPULATION_PATH = "sf_population.csv"
    else:
        raise FileNotFoundError("Nüfus CSV bulunamadı (sf_population.csv). POPULATION_PATH ile belirtin.")
else:
    if POPULATION_PATH.startswith(("http://","https://")):
        raise ValueError("CSV-ONLY: POPULATION_PATH yerel bir CSV olmalı (URL kabul edilmez).")
    if not Path(POPULATION_PATH).exists():
        raise FileNotFoundError(f"POPULATION_PATH yok: {POPULATION_PATH}")

# İsteğe bağlı hedef seviye (auto: veri uzunluğundan çıkar)
CENSUS_GEO_LEVEL = os.getenv("CENSUS_GEO_LEVEL", "auto").strip().lower()
MAP_LEN = {"county": 5, "tract": 11, "blockgroup": 12, "block": 15}

# ----------------------------- Read -----------------------------
crime = pd.read_csv(CRIME_INPUT, low_memory=False)
crime_geoid_col = _find_geoid_col(crime)
if not crime_geoid_col:
    raise RuntimeError("Suç veri setinde GEOID kolonu bulunamadı.")

# Nüfusu str okuyalım ki virgül/format bozulmasın
pop = pd.read_csv(POPULATION_PATH, low_memory=False, dtype=str)
pop_geoid_col = _find_geoid_col(pop)
if not pop_geoid_col:
    raise RuntimeError("Nüfus CSV’de GEOID kolonu bulunamadı (örn. GEOID/geography_id).")

pop_val_col = _find_population_col(pop)
if not pop_val_col:
    raise RuntimeError("Nüfus CSV’de nüfus değeri için bir kolon bulunamadı (örn. population/B01003_001E/estimate).")

# ----------------------------- Level & Keys -----------------------------
crime_len = _mode_len(_digits_only(crime[crime_geoid_col]))
pop_len   = _mode_len(_digits_only(pop[pop_geoid_col]))

if CENSUS_GEO_LEVEL in MAP_LEN:
    join_len = MAP_LEN[CENSUS_GEO_LEVEL]
else:
    # auto: veriye göre makul birleşik anahtar uzunluğu
    # Daha ince veriyi daha kaba seviyeye toplayabiliriz; county ise 5’te birleşir.
    join_len = min(max(5, crime_len), max(5, pop_len))

print(f"[info] crime GEO len≈{crime_len} | pop GEO len≈{pop_len} | join_len={join_len} ({_level_name(join_len)})")

# ----------------------------- Prep Population -----------------------------
pp = pop[[pop_geoid_col, pop_val_col]].copy()
pp["_key"] = _key(pp[pop_geoid_col], join_len)

# nüfusu numeriğe çevir (virgül/boşluk temizliği)
pp["population"] = (
    pp[pop_val_col].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(" ", "", regex=False)
)
pp["population"] = pd.to_numeric(pp["population"], errors="coerce").fillna(0)

# Eğer pop daha ince ise (ör. blockgroup 12 → join 11), aggregate et
if pop_len > join_len:
    pp = pp.groupby("_key", as_index=False)["population"].sum()
else:
    pp = pp[["_key", "population"]].drop_duplicates("_key")

# ----------------------------- Prep Crime & Merge -----------------------------
cc = crime.copy()
cc["_key"] = _key(cc[crime_geoid_col], join_len)

# Hızlı kalite göstergesi
ok_pop   = _len_ok(pp["_key"], join_len)
ok_crime = _len_ok(cc["_key"], join_len)
print(f"🔎 GEO normalize: level={_level_name(join_len)} (L={join_len}) | pop_ok={ok_pop:.2%} | crime_ok={ok_crime:.2%}")

out = cc.merge(pp, how="left", on="_key", suffixes=("", "_demog"))
out.drop(columns=["_key"], errors="ignore", inplace=True)

# ----------------------------- Save & Logs -----------------------------
Path(CRIME_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(CRIME_OUTPUT, index=False)
print(f"✅ Kaydedildi → {CRIME_OUTPUT}")

try:
    print(f"📊 satır: crime={len(crime):,} | pop={len(pp):,} | out={len(out):,}")
    with pd.option_context("display.max_columns", 50, "display.width", 2000):
        print(out[[crime_geoid_col, "population"]].head(5).to_string(index=False))
except Exception as e:
    print(f"ℹ️ Önizleme atlandı: {e}")

if __name__ == "__main__":
    pass
