# update_911.py
from __future__ import annotations
import os, re, io, time, requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

# zoneinfo güvenli import (SF saat dilimi), olmazsa zarifçe devre dışı
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

# =========================
# LOG / HELPERS
# =========================

def log(msg: str):
    print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        log(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        log(f"📁 Yedek oluşturuldu: {path}.bak")

def _to_date_series(x):
    """UTC -> SF yerel tarihe dönüştürmeyi dener; olmazsa naive tarihe düşer."""
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce")
        if SF_TZ is not None:
            s = s.dt.tz_convert(SF_TZ)
        return s.dt.date.dropna()
    except Exception:
        return pd.to_datetime(x, errors="coerce").dt.date.dropna()

def log_shape(df, label):
    r, c = df.shape
    log(f"📊 {label}: {r} satır × {c} sütun")

def log_date_range(df, date_col="date", label="911"):
    if date_col not in df.columns:
        log(f"⚠️ {label}: '{date_col}' kolonu yok.")
        return
    s = _to_date_series(df[date_col])
    if s.empty:
        log(f"⚠️ {label}: tarih parse edilemedi.")
        return
    log(f"🧭 {label} tarihi aralığı: {s.min()} → {s.max()} (gün={s.nunique()})")

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    """Sadece rakamları al, soldan L karaktere kes ve zfill(L) yap (panel ile uyumlu)."""
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def is_lfs_pointer_file(p: Path) -> bool:
    """Git LFS pointer dosyası olup olmadığını hızlıca kontrol et."""
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False

# =========================
# CONFIG & PATHS
# =========================

# ✅ Panel ile tutarlılık: varsayılan GEOID uzunluğu 11
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# Çalışma dizini ENV ile yönetilebilir
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# 911 summary dosya adları
LOCAL_NAME = "sf_911_last_5_year.csv"
local_summary_path = Path(BASE_DIR) / LOCAL_NAME
Y_NAME = "sf_911_last_5_year_y.csv"
y_summary_path = Path(BASE_DIR) / Y_NAME

# Crime grid
CRIME_GRID_CANDIDATES = [
    Path(BASE_DIR) / "sf_crime_grid_full_labeled.csv",
    Path("./sf_crime_grid_full_labeled.csv"),
]
merged_output_path = Path(BASE_DIR) / "sf_crime_01.csv"

# Census blocks (komşu için)
CENSUS_CANDIDATES = [
    Path(BASE_DIR) / "sf_census_blocks_with_population.geojson",
    Path("./sf_census_blocks_with_population.geojson"),
]

# API / kaynak
SF911_API_URL   = os.getenv("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF_APP_TOKEN    = os.getenv("SF911_API_TOKEN", "")
AGENCY_FILTER   = os.getenv("SF911_AGENCY_FILTER", "agency like '%Police%'")
REQUEST_TIMEOUT = int(os.getenv("SF911_REQUEST_TIMEOUT", "60"))
CHUNK_LIMIT     = int(os.getenv("SF911_CHUNK_LIMIT", "50000"))
MAX_RETRIES     = int(os.getenv("SF911_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS = float(os.getenv("SF911_SLEEP", "0.2"))
BULK_RANGE      = os.getenv("SF911_BULK_RANGE", "1").lower() in ("1","true","yes","on")
IS_V3           = "/api/v3/views/" in SF911_API_URL
V3_PAGE_LIMIT   = int(os.getenv("SF_V3_PAGE_LIMIT", "1000"))
SF911_RECENT_HOURS = int(os.getenv("SF911_RECENT_HOURS", "6"))

# Release taban URL — `_y` ÖNCELİKLİ, sonra eski ada düş
RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()
RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]

def _pick_working_release_url(candidates: list[str]) -> str:
    """
    Aday release URL'lerini sırayla dener; erişilebilir ve LFS pointer olmayan
    ilkini döndürür. Hiçbiri olmazsa RuntimeError fırlatır.
    """
    for u in candidates:
        if not u:
            continue
        try:
            r = requests.get(u, timeout=20)
            # Git LFS pointer'ları genelde çok küçük olur ve başında 'git-lfs' geçer
            if r.ok and r.content and len(r.content) > 200 and b"git-lfs" not in r.content[:200].lower():
                log(f"⬇️ Release kaynağı seçildi: {u}")
                return u
            else:
                log(f"⚠️ Uygun değil (boş/küçük/LFS pointer olabilir): {u}")
        except Exception as e:
            log(f"⚠️ Ulaşılamadı: {u} ({e})")
    raise RuntimeError("❌ Hiçbir release 911 URL’i erişilebilir değil.")

# Komşu ayarları
ENABLE_NEIGHBORS  = os.getenv("ENABLE_NEIGHBORS", "1").lower() in ("1","true","yes","on")
NEIGHBOR_METHOD   = os.getenv("NEIGHBOR_METHOD", "touches")  # touches | radius
NEIGHBOR_RADIUS_M = float(os.getenv("NEIGHBOR_RADIUS_M", "500"))

# BBOX temizliği
SF_BBOX = (-123.2, 37.6, -122.3, 37.9)

# Zaman pencereleri (rolling)
ROLL_WINDOWS = (3, 7)

# --- KOMBİNE KAYIT HELPER ---
def save_911_both(df: pd.DataFrame):
    """911 özetini hem normal ada hem _y ada kaydet."""
    safe_save_csv(df, str(local_summary_path))
    safe_save_csv(df, str(y_summary_path))
    log(f"💾 911 özet yazıldı → {local_summary_path.name} & {y_summary_path.name} (satır={len(df)})")

# =========================
# IO HELPERS
# =========================

def read_large_csv_in_chunks(path, usecols=None, chunksize=200_000):
    try:
        it = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"}, usecols=usecols, chunksize=chunksize)
        return pd.concat(it, ignore_index=True)
    except ValueError:
        it = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"}, chunksize=chunksize)
        return pd.concat(it, ignore_index=True)

# =========================
# 911 FETCH & SUMMARY
# =========================

def try_small_request(params, headers):
    p = dict(params)
    p["$limit"], p["$offset"] = 1, 0
    r = requests.get(SF911_API_URL, headers=headers, params=p, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r

def _load_blocks() -> tuple[gpd.GeoDataFrame, int]:
    census_path = next((p for p in CENSUS_CANDIDATES if p.exists()), None)
    if census_path is None:
        raise FileNotFoundError("❌ Nüfus blokları GeoJSON yok (crime_prediction_data/ veya kök).")
    gdf_blocks = gpd.read_file(census_path)
    if "GEOID" not in gdf_blocks.columns:
        cand = [c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID benzeri bir sütun yok.")
        gdf_blocks = gdf_blocks.rename(columns={cand[0]: "GEOID"})
    tlen = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
    gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], tlen)
    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True)
    elif gdf_blocks.crs.to_epsg() != 4326:
        gdf_blocks = gdf_blocks.to_crs(4326)
    return gdf_blocks, tlen

def ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df
    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "intersection_point" in df.columns:
            def _lon(x):
                if isinstance(x, dict) and "coordinates" in x: return x["coordinates"][0]
                if isinstance(x, str):
                    m = re.search(r"[-\d\.]+,\s*[-\d\.]+", x)
                    if m:
                        lo, la = m.group(0).split(","); return float(lo)
                return None
            def _lat(x):
                if isinstance(x, dict) and "coordinates" in x: return x["coordinates"][1]
                if isinstance(x, str):
                    m = re.search(r"[-\d\.]+,\s*[-\d\.]+", x)
                    if m:
                        lo, la = m.group(0).split(","); return float(la)
                return None
            df["longitude"], df["latitude"] = df["intersection_point"].apply(_lon), df["intersection_point"].apply(_lat)
        for a,b in (("y","x"),("lat","long")):
            if a in df.columns and b in df.columns and "latitude" not in df.columns:
                df["latitude"], df["longitude"] = pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")
                break
    if "latitude" in df.columns and "longitude" in df.columns:
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df = df[(df["latitude"].between(min_lat, max_lat)) & (df["longitude"].between(min_lon, max_lon))]
    df = df.dropna(subset=["latitude","longitude"]).copy()

    gdf_blocks, tlen = _load_blocks()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID","geometry"]], how="left", predicate="within")
    out = pd.DataFrame(gdf.drop(columns=["geometry","index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    out = out.dropna(subset=["GEOID"]).copy()
    return out

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["GEOID","date","hour_range","911_request_count_hour_range","911_request_count_daily(before_24_hours)"])
    df = raw.copy()
    ts_col = None
    for cand in ["received_time","received_datetime","date","datetime","timestamp","call_received_datetime"]:
        if cand in df.columns:
            ts_col = cand; break
    if ts_col is None:
        raise ValueError("Zaman kolonu bulunamadı (received_time/received_datetime/date).")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour
    eh = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int) % 24
    start = (eh // 3) * 3
    df["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")

    has_geoid = "GEOID" in df.columns
    grp_hr  = (["GEOID"] if has_geoid else []) + ["date","hour_range"]
    grp_day = (["GEOID"] if has_geoid else []) + ["date"]

    hr_agg = df.groupby(grp_hr, dropna=False, observed=True).size().reset_index(name="911_request_count_hour_range")
    day_agg = df.groupby(grp_day, dropna=False, observed=True).size().reset_index(name="911_request_count_daily(before_24_hours)")
    out = hr_agg.merge(day_agg, on=grp_day, how="left")

    cols_tail = [c for c in ["date","hour_range","GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]

# -------------------- LOCAL BASE (artifact/regular) & RELEASE ------------------

def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 tabanı okunuyor: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID":"string"})
    is_already_summary = (
        {"date","hour_range"}.issubset(df.columns) and
        any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    )
    if is_already_summary:
        cnt_col = next(c for c in ["911_request_count_hour_range","call_count","count","requests","n"] if c in df.columns)
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})
        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m: return None
            a = int(m.group(1)) % 24; b = int(m.group(2)); b = b if b > a else min(a+3, 24)
            return f"{a:02d}-{b:02d}"
        df["hour_range"] = df["hour_range"].apply(_fmt_hr)
        if "911_request_count_daily(before_24_hours)" not in df.columns:
            keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
            day = df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"] \
                    .sum().reset_index(name="911_request_count_daily(before_24_hours)")
            df = df.merge(day, on=keys, how="left")
        if min_date is not None:
            df = df[df["date"] >= min_date]
        cols_tail = [c for c in ["date","hour_range","GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in cols_tail] + cols_tail
        return df[cols]
    # değilse ham → özet
    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date]
    return std

def ensure_local_911_base() -> Optional[Path]:
    """
    Tercih sırası:
      1) artifact/çalışma alanından sf_911_last_5_year_y.csv (önceki full pipeline’dan)
      2) yerel sf_911_last_5_year.csv
      3) yoksa None (release fallback kullanılacak)
    """
    y_candidates = [
        Path(BASE_DIR) / "sf_911_last_5_year_y.csv",
        Path("./sf_911_last_5_year_y.csv"),
        Path("outputs/sf_911_last_5_year_y.csv"),
    ]
    for p in y_candidates:
        if p.exists():
            if p.suffix == ".csv" and is_lfs_pointer_file(p):
                continue
            log(f"📦 911 base (preferred Y) bulundu: {p}")
            return p

    regular_candidates = [
        Path(BASE_DIR) / "sf_911_last_5_year.csv",
        Path("./sf_911_last_5_year.csv"),
    ]
    for p in regular_candidates:
        if p.exists():
            if p.suffix == ".csv" and is_lfs_pointer_file(p):
                continue
            log(f"📦 911 base (regular) bulundu: {p}")
            return p

    return None

# ------------------------- RELEASE → BASE SUMMARY ----------------------------

def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release 911 özeti indiriliyor: {url}")
    r = requests.get(url, timeout=120); r.raise_for_status()
    tmp = Path(BASE_DIR) / "_tmp_911.csv"; ensure_parent(str(tmp))
    tmp.write_bytes(r.content)
    df = pd.read_csv(tmp, low_memory=False, dtype={"GEOID":"string"})
    is_already_summary = ( {"date","hour_range"}.issubset(df.columns)
                           and any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"]) )
    if is_already_summary:
        cnt_col = next(c for c in ["911_request_count_hour_range","call_count","count","requests","n"] if c in df.columns)
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})
        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        # hour_range normalize
        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m: return None
            a = int(m.group(1)) % 24; b = int(m.group(2)); b = b if b > a else min(a+3, 24)
            return f"{a:02d}-{b:02d}"
        df["hour_range"] = df["hour_range"].apply(_fmt_hr)
        if "911_request_count_daily(before_24_hours)" not in df.columns:
            keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
            day = df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"].sum().reset_index(name="911_request_count_daily(before_24_hours)")
            df = df.merge(day, on=keys, how="left")
        if min_date is not None:
            df = df[df["date"] >= min_date]
        cols_tail = [c for c in ["date","hour_range","GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in cols_tail] + cols_tail
        return df[cols]
    # değilse ham → özet
    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date]
    return std

# ----------------------- API INCREMENT (RANGE) --------------------------------

def fetch_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    dt_candidates = ["received_time","received_datetime","date","datetime","call_datetime","received_dttm","call_date"]
    headers = {"X-App-Token": SF_APP_TOKEN} if SF_APP_TOKEN else {}
    rng_start = f"{start_day}T00:00:00"; rng_end = f"{end_day}T23:59:59"
    chosen_dt, last_err = None, None
    for dt_col in dt_candidates:
        base_where = f"{dt_col} between '{rng_start}' and '{rng_end}'"
        for wc in [base_where + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""), base_where]:
            try:
                try_small_request({"$where": wc}, headers)
                chosen_dt = dt_col; break
            except Exception as e:
                last_err = e; continue
        if chosen_dt: break
    if chosen_dt is None:
        log(f"    ❌ Aralık için uygun datetime kolonu bulunamadı. Son hata: {last_err}")
        return None
    pieces, offset, page = [], 0, 1
    where_list = [f"{chosen_dt} between '{rng_start}' and '{rng_end}'" + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""),
                  f"{chosen_dt} between '{rng_start}' and '{rng_end}'"]
    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(SF911_API_URL, headers=headers, params={"$where": where_list[0], "$limit": CHUNK_LIMIT, "$offset": offset}, timeout=REQUEST_TIMEOUT)
                if r.status_code == 400:
                    r = requests.get(SF911_API_URL, headers=headers, params={"$where": where_list[1], "$limit": CHUNK_LIMIT, "$offset": offset}, timeout=REQUEST_TIMEOUT)
                r.raise_for_status(); df = pd.read_json(io.BytesIO(r.content)); break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"    ❌ range page {page} (offset={offset}) hata: {e}")
                    df = None; break
                time.sleep(1.0 + attempt*0.5)
        if df is None or df.empty:
            if page == 1: log("    (bu aralıkta veri yok)")
            break
        log(f"    + {len(df)} satır (range-page={page}, offset={offset})")
        pieces.append(df)
        if len(df) < CHUNK_LIMIT: break
        offset += CHUNK_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)
    if not pieces: return None
    return pd.concat(pieces, ignore_index=True)

def fetch_v3_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    from requests.adapters import HTTPAdapter, Retry
    sess = requests.Session()
    retries = Retry(total=5, connect=5, read=5, backoff_factor=1.2,
                    status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://",  HTTPAdapter(max_retries=retries))

    headers = {"Accept":"application/json"}
    if SF_APP_TOKEN:
        headers["X-App-Token"] = SF_APP_TOKEN

    dt_candidates = ["received_time","received_datetime","date","datetime","call_datetime","received_dttm","call_date"]
    rng_start = f"{start_day}T00:00:00"; rng_end = f"{end_day}T23:59:59"

    chosen_dt, cols = None, None
    for dtc in dt_candidates:
        where = f"{dtc} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER:
            where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT 1 OFFSET 0"
        try:
            r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if obj.get("data"):
                chosen_dt = dtc
                cols = [c.get("fieldName") or c.get("name") or f"c{i}"
                        for i,c in enumerate(obj.get("meta",{}).get("view",{}).get("columns",[]))]
                break
        except Exception:
            continue
    if not chosen_dt:
        log("    ❌ v3: uygun datetime kolonu bulunamadı.")
        return None

    all_rows, offset, page = [], 0, 1
    while True:
        where = f"{chosen_dt} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER:
            where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT {V3_PAGE_LIMIT} OFFSET {offset}"

        got = 0
        for attempt in range(MAX_RETRIES):
            try:
                r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                obj = r.json()
                data = obj.get("data", [])
                if data:
                    for row in data:
                        if isinstance(row, list):
                            all_rows.append({cols[i]: (row[i] if i < len(cols) else None) for i in range(len(cols))})
                        elif isinstance(row, dict):
                            all_rows.append(row)
                    got = len(data)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"    ❌ v3 range page {page} (offset={offset}) hata: {e}")
                time.sleep(1.0 + attempt*0.5)
        if got < V3_PAGE_LIMIT:
            break
        offset += V3_PAGE_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)

    return pd.DataFrame(all_rows)

def write_recent_csv(raw: pd.DataFrame, hours: int = SF911_RECENT_HOURS):
    ts_col = next((c for c in ["received_time","received_datetime","date","datetime",
                               "timestamp","call_received_datetime","ts"] if c in raw.columns), None)
    if not ts_col:
        return
    tmp = raw.copy()
    tmp["ts"] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp[tmp["ts"].notna()]
    if tmp.empty:
        return

    lat_col = next((c for c in ["latitude","lat","y"] if c in raw.columns), None)
    lon_col = next((c for c in ["longitude","lon","x"] if c in raw.columns), None)

    tmax = tmp["ts"].max()
    cutoff = tmax - pd.Timedelta(hours=hours)

    out = pd.DataFrame({"ts": tmp["ts"]})
    if lat_col: out["lat"] = pd.to_numeric(tmp[lat_col], errors="coerce")
    if lon_col: out["lon"] = pd.to_numeric(tmp[lon_col], errors="coerce")
    out = out[out["ts"] >= cutoff].copy()

    path = Path(BASE_DIR) / "sf_911_recent.csv"
    safe_save_csv(out, str(path))
    log(f"ℹ️ sf_911_recent.csv yazıldı (son {hours} saat): {len(out)} satır")

def incremental_summary(start_day: datetime.date, end_day: datetime.date) -> pd.DataFrame:
    if start_day is None or end_day is None or end_day < start_day:
        return pd.DataFrame()
    log(f"🌐 API artımlı: {start_day} → {end_day} ({(end_day - start_day).days + 1} gün)")
    raw = None
    if BULK_RANGE:
        raw = fetch_v3_range_all_chunks(start_day, end_day) if IS_V3 else fetch_range_all_chunks(start_day, end_day)
    try:
        if raw is not None and not raw.empty:
            write_recent_csv(raw, hours=SF911_RECENT_HOURS)
    except Exception as e:
        log(f"⚠️ recent yazımı atlandı: {e}")
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        raw = ensure_geoid(raw)
    except Exception as e:
        log(f"⚠️ ensure_geoid sırasında hata: {e}; GEOID’siz özet üretilecek")
    return make_standard_summary(raw)

# =========================
# MAIN: LOCAL (Y/regular) → FALLBACK RELEASE + INCREMENT → ENRICH → MERGE
# =========================

five_years_ago = datetime.now(timezone.utc).date() - timedelta(days=5*365)

log(f"📁 911 yerel özet yolu: {local_summary_path}")

# 1) Önce yerel tabanı dene (artifact'tan gelen Y öncelikli)
base_csv_path = ensure_local_911_base()
if base_csv_path is not None:
    final_911 = summary_from_local(base_csv_path, min_date=five_years_ago)
    save_911_both(final_911)
    log(f"✅ Yerel 911 özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")
else:
    # 2) Release fallback (Y URL'leri öncelikli)
    release_url = _pick_working_release_url(RAW_911_URL_CANDIDATES)
    final_911 = summary_from_release(release_url, min_date=five_years_ago)
    save_911_both(final_911)
    log(f"✅ Release özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")

# 3) Max tarihten bugüne SF saatine göre artımlı aralık seç
base_max_date = to_date(final_911["date"]).max() if not final_911.empty else None

today_sf = (datetime.now(SF_TZ) if SF_TZ is not None else datetime.now()).date()
if base_max_date is None:
    fetch_start, fetch_end = today_sf, today_sf
else:
    fetch_start, fetch_end = base_max_date + timedelta(days=1), today_sf
    if fetch_start > fetch_end:
        fetch_start = fetch_end
log(f"🗓️ İndirme aralığı: {fetch_start} → {fetch_end} ({(fetch_end - fetch_start).days + 1} gün)")

# 4) Artımlı API verisini çek ve taban özetle birleştir
inc = incremental_summary(fetch_start, fetch_end)
if inc is not None and not inc.empty:
    if "GEOID" in inc.columns:
        inc["GEOID"] = normalize_geoid(inc["GEOID"], DEFAULT_GEOID_LEN)
    inc["date"] = to_date(inc["date"])
    before = len(final_911)
    final_911 = pd.concat([final_911, inc], ignore_index=True)
    subset_cols = [c for c in ["GEOID","date","hour_range"] if c in final_911.columns]
    final_911 = (final_911.dropna(subset=["date"])
                             .sort_values(subset_cols if subset_cols else ["date"])
                             .drop_duplicates(subset=subset_cols if subset_cols else ["date"], keep="last"))
    final_911 = final_911[final_911["date"] >= five_years_ago]
    save_911_both(final_911)
    log(f"💾 911 özet GÜNCELLENDİ (base+API) → {local_summary_path} & {y_summary_path} (+{len(final_911)-before:,} satır)")
else:
    log("ℹ️ API tarafında yeni gün yok veya boş döndü; taban veri geçerli.")

if final_911 is None or final_911.empty:
    log("⚠️ 911 özeti üretilemedi (boş). Çıkılıyor.")
    raise SystemExit(0)

# =========================
# STANDARDIZE + DERIVED KEYS (hr_key, dow, season)
# =========================

hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")

def _hr_key_from_range(hr):
    m = hr_pat.match(str(hr))
    return int(m.group(1)) % 24 if m else None

final_911 = final_911.dropna(subset=["GEOID","date","hour_range"]).copy()
final_911["GEOID"] = normalize_geoid(final_911["GEOID"], DEFAULT_GEOID_LEN)
final_911["date"] = to_date(final_911["date"])
final_911["hr_key"] = final_911["hour_range"].apply(_hr_key_from_range).astype("int16")
final_911["day_of_week"] = pd.to_datetime(final_911["date"]).dt.weekday.astype("int8")
final_911["month"] = pd.to_datetime(final_911["date"]).dt.month.astype("int8")
_season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
final_911["season"] = final_911["month"].map(_season_map).astype("category")

log_shape(final_911, "911 summary (normalize)")
log_date_range(final_911, "date", "911")

# =========================
# ROLLING (3g/7g) — GEOID ve GEOID×hr_key
# =========================

_day_unique = (final_911[["GEOID","date","911_request_count_daily(before_24_hours)"]]
               .drop_duplicates(subset=["GEOID","date"]))
_day_unique = _day_unique.sort_values(["GEOID","date"]).rename(columns={"911_request_count_daily(before_24_hours)":"daily_cnt"}).reset_index(drop=True)

_hr_unique = (final_911[["GEOID","hr_key","date","911_request_count_hour_range"]]
              .groupby(["GEOID","hr_key","date"], as_index=False, observed=True)["911_request_count_hour_range"].sum())
_hr_unique = _hr_unique.rename(columns={"911_request_count_hour_range":"hr_cnt"}).sort_values(["GEOID","hr_key","date"]).reset_index(drop=True)

for W in ROLL_WINDOWS:
    _day_unique[f"911_geo_last{W}d"] = (
        _day_unique.groupby("GEOID")["daily_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
    ).astype("float32")
    _hr_unique[f"911_geo_hr_last{W}d"] = (
        _hr_unique.groupby(["GEOID","hr_key"])["hr_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
    ).astype("float32")

# =========================
# KOMŞU GEOID ÖZELLİKLERİ (günlük baz)
# =========================

def build_neighbors(method: str = "touches", radius_m: float = 500.0) -> pd.DataFrame:
    gdf_blocks, _ = _load_blocks()
    tracts = gdf_blocks.dissolve(by="GEOID", as_index=False)
    if method == "radius":
        tr_utm = tracts.to_crs("EPSG:26910")
        buf = tr_utm.buffer(radius_m)
        g_buf = gpd.GeoDataFrame(tr_utm[["GEOID"]].copy(), geometry=buf, crs=tr_utm.crs)
        join = gpd.sjoin(g_buf, tr_utm[["GEOID","geometry"]].rename(columns={"GEOID":"nbr"}), predicate="intersects")
        edges = join[["GEOID","nbr"]]
    else:
        join = gpd.sjoin(tracts[["GEOID","geometry"]], tracts[["GEOID","geometry"]].rename(columns={"GEOID":"nbr"}), predicate="touches")
        edges = join[["GEOID","nbr"]]
    edges = edges[edges["GEOID"] != edges["nbr"]].copy()
    edges["pair"] = edges.apply(lambda r: tuple(sorted((r["GEOID"], r["nbr"]))), axis=1)
    edges = edges.drop_duplicates("pair").drop(columns=["pair"])
    edges["GEOID"] = normalize_geoid(edges["GEOID"], DEFAULT_GEOID_LEN)
    edges["nbr"] = normalize_geoid(edges["nbr"], DEFAULT_GEOID_LEN)
    return pd.DataFrame(edges)

neighbors_df = None
if ENABLE_NEIGHBORS:
    try:
        neighbors_df = build_neighbors(NEIGHBOR_METHOD, NEIGHBOR_RADIUS_M)
        log_shape(neighbors_df, f"Komşu haritası ({NEIGHBOR_METHOD})")
    except Exception as e:
        log(f"⚠️ Komşu haritası üretilemedi: {e}")
        neighbors_df = None

_neighbor_roll = None
if neighbors_df is not None and not neighbors_df.empty:
    day_nbr = neighbors_df.merge(_day_unique.rename(columns={"GEOID":"nbr"}), on="nbr", how="left")
    day_nbr = day_nbr.groupby(["GEOID","date"], as_index=False, observed=True)["daily_cnt"].sum().rename(columns={"daily_cnt":"nbr_daily_cnt"})
    _neighbor_roll = day_nbr.sort_values(["GEOID","date"]).reset_index(drop=True)
    for W in ROLL_WINDOWS:
        _neighbor_roll[f"911_neighbors_last{W}d"] = (
            _neighbor_roll.groupby("GEOID")["nbr_daily_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

# =========================
# MERGE STRATEJİSİ
# =========================

_enriched = final_911.merge(_hr_unique, on=["GEOID","hr_key","date"], how="left")
_enriched = _enriched.merge(_day_unique, on=["GEOID","date"], how="left")
if _neighbor_roll is not None:
    _enriched = _enriched.merge(_neighbor_roll[["GEOID","date"] + [f"911_neighbors_last{W}d" for W in ROLL_WINDOWS]], on=["GEOID","date"], how="left")

crime_grid_path = next((p for p in CRIME_GRID_CANDIDATES if p.exists()), None)
if crime_grid_path is None:
    raise FileNotFoundError("❌ Suç grid yok: crime_prediction_data/sf_crime_grid_full_labeled.csv (veya kökte).")
crime = pd.read_csv(crime_grid_path, dtype={"GEOID": str}, low_memory=False)
log(f"📥 Suç grid yüklendi: {len(crime)} satır ({crime_grid_path})")
log_shape(crime, "CRIME grid — ham")

crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)

if "event_hour" not in crime.columns:
    raise ValueError("❌ Suç grid dosyasında 'event_hour' yok.")
crime["hr_key"] = ((pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3

has_date_col = ("date" in crime.columns) or ("datetime" in crime.columns)
if has_date_col:
    if "date" not in crime.columns:
        crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
    else:
        crime["date"] = to_date(crime["date"])
    keys = ["GEOID","date","hr_key"]
    merged = crime.merge(_enriched, on=keys, how="left")
    log("🔗 Join modu: DATE-BASED (GEOID, date, hr_key)")
else:
    cal_keys = ["GEOID","hr_key","day_of_week","season"]
    agg_cols = [
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "911_geo_last3d","911_geo_last7d",
        "911_geo_hr_last3d","911_geo_hr_last7d",
    ] + ([f"911_neighbors_last{W}d" for W in ROLL_WINDOWS] if _neighbor_roll is not None else [])
    cal_agg = (_enriched.groupby(cal_keys, as_index=False, observed=True)[agg_cols]
                        .median(numeric_only=True))
    if "day_of_week" not in crime.columns:
        log("ℹ️ crime grid’de day_of_week yok → 0 atanıyor (düşük etkili).")
        crime["day_of_week"] = 0
    if "season" not in crime.columns:
        if "month" in crime.columns:
            _smap = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
            crime["season"] = crime["month"].map(_smap).fillna("Summer")
        else:
            crime["season"] = "Summer"
    merged = crime.merge(cal_agg, on=cal_keys, how="left")
    log("🔗 Join modu: CALENDAR-BASED (GEOID, hr_key, day_of_week, season)")

fill_cols = [
    "911_request_count_hour_range",
    "911_request_count_daily(before_24_hours)",
    "911_geo_last3d","911_geo_last7d",
    "911_geo_hr_last3d","911_geo_hr_last7d",
] + ([f"911_neighbors_last{W}d" for W in ROLL_WINDOWS] if _neighbor_roll is not None else [])
for c in fill_cols:
    if c in merged.columns:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("int32")

safe_save_csv(merged, str(merged_output_path))
log_shape(merged, "CRIME⨯911 (kayıt öncesi)")
log(f"✅ Suç + 911 birleştirmesi tamamlandı → {merged_output_path}")

try:
    print(merged.head(5).to_string(index=False))
except Exception:
    pass
