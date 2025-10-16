# scripts/update_weather.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pandas as pd

# --- LOG HELPERS: tarih aralığı + shape + delta --------------------------------
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

def _to_date_series(x):
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce")
        if SF_TZ is not None:
            s = s.dt.tz_convert(SF_TZ)
        s = s.dt.date
    except Exception:
        s = pd.to_datetime(x, errors="coerce").dt.date
    return pd.Series(s).dropna()

def log_shape(df: pd.DataFrame, label: str) -> None:
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_date_range(df: pd.DataFrame, date_col: str = "date", label: str = "Veri") -> None:
    if date_col not in df.columns:
        print(f"⚠️ {label}: '{date_col}' kolonu yok.")
        return
    s = _to_date_series(df[date_col])
    if s.empty:
        print(f"⚠️ {label}: tarih parse edilemedi.")
        return
    print(f"🧭 {label} tarihi aralığı: {s.min()} → {s.max()} (gün={s.nunique()})")

def log_delta(before_shape, after_shape, label: str) -> None:
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

# --- Yardımcılar ---------------------------------------------------------------
def ensure_parent(path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str) -> None:
    """Atomic yazım: tmp → replace; hata halinde .bak bırak."""
    ensure_parent(path)
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"💾 Kaydedildi: {path}")
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        try:
            df.to_csv(path + ".bak", index=False)
            print(f"📁 Yedek oluşturuldu: {path}.bak")
        except Exception:
            pass

def find_col(cols, candidates):
    m = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    return None

# --- GEOID helpers -------------------------------------------------------------
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

def normalize_geoid(series: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def normalize_geoid_tract11(s: pd.Series) -> pd.Series:
    return normalize_geoid(s, 11)

# --- Yol bulucu + ENV override + LFS uyarısı ----------------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
CWD = Path.cwd()

def _size_hint(p: Path) -> str:
    try:
        return f"{p.stat().st_size} bytes"
    except Exception:
        return "n/a"

def _env_or_none(key: str):
    v = os.getenv(key, "").strip()
    return v or None

print(f"📂 CWD  = {CWD}")
print(f"📂 ROOT = {ROOT}")

SEARCH_DIRS = [
    CWD,
    ROOT,
    CWD / "crime_prediction_data",
    ROOT / "crime_prediction_data",
    ROOT.parent / "crime_prediction_data",
]

def _expand_candidates(names):
    cands, seen = [], set()
    for d in SEARCH_DIRS:
        for n in names:
            p = d / n
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                cands.append(p)
    return cands

def pick_existing(paths):
    for p in paths:
        pth = Path(p)
        if pth.exists():
            try:
                if pth.suffix == ".csv" and pth.stat().st_size < 200:
                    print(f"⚠️ Olası LFS pointer: {pth} ({_size_hint(pth)}) → git lfs pull önerilir.")
            except Exception:
                pass
            return str(pth)
    return None

# ============== 1) Dosya yolları / girişler ===================================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(exist_ok=True)

# ZORUNLU: 07 girişi (ara dosyalar _y ALMAZ)
CRIME_INPUT_NAMES = ["sf_crime_07.csv"]  # sadece 07 kabul

# HAVA KAYNAĞI: ÖNCELİKLE _y, yoksa eski ad
# ENV ile tek dosya işaretlemek istersen: WEATHER_PATH=/full/path/to/sf_weather_5years_y.csv
WEATHER_PATH_ENV = _env_or_none("WEATHER_PATH")
if WEATHER_PATH_ENV:
    WEATHER_NAMES = [WEATHER_PATH_ENV]
else:
    WEATHER_NAMES = ["sf_weather_5years_y.csv", "sf_weather_5years.csv"]

# Grid (Y_label backfill kaynağı) — kalıcı master
GRID_NAMES = ["sf_crime_grid_full_labeled.csv"]

CRIME_INPUT_CANDS   = _expand_candidates(CRIME_INPUT_NAMES)
WEATHER_INPUT_CANDS = _expand_candidates(WEATHER_NAMES)
GRID_INPUT_CANDS    = _expand_candidates(GRID_NAMES)

crime_path   = pick_existing([str(p) for p in CRIME_INPUT_CANDS])
weather_path = pick_existing([str(p) for p in WEATHER_INPUT_CANDS])
grid_path    = pick_existing([str(p) for p in GRID_INPUT_CANDS])

if not crime_path:
    raise FileNotFoundError("❌ Zorunlu giriş bulunamadı: sf_crime_07.csv")

ALLOW_STUB = os.getenv("ALLOW_STUB_ON_API_FAIL", "1").strip().lower() not in ("0", "false")

# ============== 2) Verileri yükle + Y_label backfill ==========================
print("📥 Veriler yükleniyor...")
df_crime = pd.read_csv(crime_path, low_memory=False)
log_shape(df_crime, "CRIME (07 — ham)")

if "GEOID" not in df_crime.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok (07).")
df_crime["GEOID"] = normalize_geoid_tract11(df_crime["GEOID"])

# --- Y_label BACKFILL (gerekirse)
if "Y_label" not in df_crime.columns or df_crime["Y_label"].isna().all():
    if not grid_path:
        raise FileNotFoundError(
            "❌ 07 içinde Y_label yok ve 'sf_crime_grid_full_labeled.csv' bulunamadı (backfill için gerekli)."
        )
    print("🧩 Y_label backfill başlıyor → sf_crime_grid_full_labeled.csv")
    grid = pd.read_csv(grid_path, low_memory=False)

    if "GEOID" not in grid.columns:
        raise KeyError("❌ Grid dosyasında 'GEOID' yok.")
    grid["GEOID"] = normalize_geoid_tract11(grid["GEOID"])

    def _hr_from_hour(h):
        try:
            h = int(pd.to_numeric(h, errors="coerce"))
            return int((h // 3) * 3) % 24
        except Exception:
            return np.nan

    if "hr_key" not in df_crime.columns and "event_hour" in df_crime.columns:
        df_crime["hr_key"] = df_crime["event_hour"].apply(_hr_from_hour).astype("Int64")
    if "hr_key" not in grid.columns and "event_hour" in grid.columns:
        grid["hr_key"] = grid["event_hour"].apply(_hr_from_hour).astype("Int64")

    keys1 = [k for k in ["GEOID","date","hr_key"] if k in df_crime.columns and k in grid.columns]
    keys2 = [k for k in ["GEOID","day_of_week","season","hr_key"] if k in df_crime.columns and k in grid.columns]

    if keys1 and {"Y_label"}.issubset(set(grid.columns)):
        df_crime["date"] = pd.to_datetime(df_crime["date"], errors="coerce").dt.date if "date" in df_crime.columns else df_crime.get("date")
        grid["date"]     = pd.to_datetime(grid["date"], errors="coerce").dt.date     if "date" in grid.columns     else grid.get("date")
        before = df_crime.shape
        df_crime = df_crime.merge(grid[keys1 + ["Y_label"]].drop_duplicates(keys1), on=keys1, how="left")
        log_delta(before, df_crime.shape, "Y_label backfill (date-based)")
    elif keys2 and {"Y_label"}.issubset(set(grid.columns)):
        before = df_crime.shape
        df_crime = df_crime.merge(grid[keys2 + ["Y_label"]].drop_duplicates(keys2), on=keys2, how="left")
        log_delta(before, df_crime.shape, "Y_label backfill (calendar-based)")
    else:
        raise RuntimeError("❌ Y_label backfill için ortak anahtarlar bulunamadı (hr_key + date veya day_of_week+season).")

    if "Y_label" not in df_crime.columns:
        raise RuntimeError("❌ Backfill sonrası da Y_label yok.")
    miss = int(df_crime["Y_label"].isna().sum())
    print(f"🧪 Y_label backfill tamam: eksik={miss}")

# ============== 3) Hava verisini yükle / STUB + kanonik kopyaları yaz =========
if weather_path:
    print(f"🌤️ Hava kaynağı: {weather_path}")
    df_weather = pd.read_csv(weather_path, low_memory=False)
else:
    if not ALLOW_STUB:
        raise FileNotFoundError("❌ Hava dosyası yok: sf_weather_5years_y.csv veya sf_weather_5years.csv (repo/crime_prediction_data).")
    print("⚠️ Hava dosyası bulunamadı → STUB (boş çerçeve) kullanılacak.")
    df_weather = pd.DataFrame(columns=["date", "TMAX", "TMIN", "PRCP"])

log_shape(df_weather, "WEATHER (ham)")
# artifacts’ta eksik kalmasın diye iki kanonik kopyayı HER KOŞULDA yaz
WEATHER_CANON_Y = os.path.join(BASE_DIR, "sf_weather_5years_y.csv")
WEATHER_CANON   = os.path.join(BASE_DIR, "sf_weather_5years.csv")
try:
    safe_save_csv(df_weather, WEATHER_CANON_Y)
    safe_save_csv(df_weather, WEATHER_CANON)
except Exception as e:
    print(f"⚠️ Kanonik weather kopyaları yazılamadı: {e}")

# ============== 4) Hava sütunlarını saptama ve birim normalizasyonu ===========
def to_celsius(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.abs().median(skipna=True)
    if pd.notna(m) and m > 80:   # 0.1°C ölçeği ise
        s = s / 10.0
    return s

def to_mm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mx = s.max(skipna=True)
    if pd.notna(mx) and mx > 200:  # GHCND PRCP 0.1mm
        s = s / 10.0
    return s

tmax_col = find_col(df_weather.columns, ["TMAX", "tmax"])
tmin_col = find_col(df_weather.columns, ["TMIN", "tmin"])
prcp_col = find_col(df_weather.columns, ["PRCP", "prcp"])

dfw = df_weather.copy()
dfw["temp_max"]         = to_celsius(dfw[tmax_col]) if tmax_col else np.nan
dfw["temp_min"]         = to_celsius(dfw[tmin_col]) if tmin_col else np.nan
dfw["temp_range"]       = dfw["temp_max"] - dfw["temp_min"]
dfw["precipitation_mm"] = to_mm(dfw[prcp_col]) if prcp_col else np.nan

for c in ["temp_max","temp_min","temp_range","precipitation_mm"]:
    dfw[c] = pd.to_numeric(dfw[c], errors="coerce")

dfw["temp_max"]         = dfw["temp_max"].clip(-60, 60)
dfw["temp_min"]         = dfw["temp_min"].clip(-60, 60)
dfw["temp_range"]       = dfw["temp_range"].clip(lower=0)
dfw["precipitation_mm"] = dfw["precipitation_mm"].clip(lower=0)

# ============== 5) GEOID bazında tek satırlık hava özeti (tarih KULLANMADAN) ==
weather_summary = {
    "temp_max":         dfw["temp_max"].median(skipna=True),
    "temp_min":         dfw["temp_min"].median(skipna=True),
    "temp_range":       dfw["temp_range"].median(skipna=True),
    "precipitation_mm": dfw["precipitation_mm"].median(skipna=True),
}

def _clip(v, lo=None, hi=None):
    try:
        if v is None or pd.isna(v): return np.nan
        if lo is not None: v = max(lo, v)
        if hi is not None: v = min(hi, v)
        return v
    except Exception:
        return np.nan

weather_summary["temp_max"]         = _clip(weather_summary["temp_max"], lo=-60, hi=60)
weather_summary["temp_min"]         = _clip(weather_summary["temp_min"], lo=-60, hi=60)
weather_summary["temp_range"]       = _clip(max(0, weather_summary["temp_range"]), lo=0)
weather_summary["precipitation_mm"] = _clip(max(0, weather_summary["precipitation_mm"]), lo=0)

print("🌤️ GEOID’e yayınlanacak hava özeti (genel medyan, date kullanılmadan):", weather_summary)

geoid_df = pd.DataFrame({"GEOID": df_crime["GEOID"].dropna().unique()})
for col, val in weather_summary.items():
    geoid_df[col] = val
log_shape(geoid_df, "WEATHER (GEOID yayın tablosu)")

# ============== 6) Sadece GEOID ile birleştir (07 → 08) =======================
_before = df_crime.shape
df_merged = df_crime.merge(geoid_df, on="GEOID", how="left")
log_delta(_before, df_merged.shape, "CRIME ⨯ WEATHER(GEOID) merge")
log_shape(df_merged, "CRIME (weather enrich sonrası)")

_na = df_merged[["temp_max","temp_min","temp_range","precipitation_mm"]].isna().sum()
print(f"🧪 Merge sonrası NA (hava sütunları): {_na.to_dict()}")

# ============== 7) Kaydet + Önizleme ==========================================
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_08.csv")  # ara dosya: _y YOK
safe_save_csv(df_merged, CRIME_OUTPUT)
print(f"✅ Hava durumu eklendi (yalnızca GEOID üzerinden) → {CRIME_OUTPUT}")
print("📄 Eklenen sütunlar:", ["temp_max","temp_min","temp_range","precipitation_mm"])
print(f"📊 Satır sayısı: {df_merged.shape[0]}, Sütun sayısı: {df_merged.shape[1]}")

try:
    preview = pd.read_csv(CRIME_OUTPUT, nrows=3, low_memory=False)
    print(f"📄 {CRIME_OUTPUT} — ilk 3 satır (tüm sütunlar):")
    with pd.option_context('display.max_columns', None, 'display.width', 2000):
        print(preview.to_string(index=False))
except Exception as e:
    print(f"⚠️ Önizleme okunamadı: {e}")
