# update_bus.py

import os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy.spatial import cKDTree

# ========== küçük yardımcılar ==========
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
def normalize_geoid(s: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def freedman_diaconis_bin_count(data: np.ndarray, max_bins: int = 10) -> int:
    data = np.asarray(data)
    if len(data) < 2 or np.all(data == data[0]):
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    bw = 2 * iqr / (len(data) ** (1 / 3))
    if bw <= 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    return max(2, min(max_bins, int(np.ceil((data.max() - data.min()) / bw))))

def extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Farklı şemalardan lat/lon çıkarır:
      - (latitude, longitude) / (lat, long) / (y, x)
      - (stop_lat, stop_lon) / (stop_latitude, stop_longitude)
      - (position_latitude, position_longitude)
      - location (JSON) / the_geom (GeoJSON)
    """
    df = df.copy()
    candidates = [
        ("latitude", "longitude"),
        ("lat", "long"),
        ("y", "x"),
        ("stop_lat", "stop_lon"),
        ("stop_latitude", "stop_longitude"),
        ("position_latitude", "position_longitude"),
    ]
    for la, lo in candidates:
        if la in df.columns and lo in df.columns:
            df["stop_lat"] = pd.to_numeric(df[la], errors="coerce")
            df["stop_lon"] = pd.to_numeric(df[lo], errors="coerce")
            return df

    if "location" in df.columns:
        def _g(o, k):
            if isinstance(o, dict):
                return o.get(k)
            if isinstance(o, str):
                try:
                    j = json.loads(o); return j.get(k)
                except Exception:
                    return None
            return None
        df["stop_lat"] = pd.to_numeric(df["location"].apply(lambda o: _g(o, "latitude")), errors="coerce")
        df["stop_lon"] = pd.to_numeric(df["location"].apply(lambda o: _g(o, "longitude")), errors="coerce")
        if df["stop_lat"].notna().any() and df["stop_lon"].notna().any():
            return df

    if "the_geom" in df.columns:
        def _coords(o):
            if isinstance(o, dict) and "coordinates" in o and isinstance(o["coordinates"], (list, tuple)) and len(o["coordinates"]) >= 2:
                lon, lat = o["coordinates"][:2]; return lat, lon
            if isinstance(o, str):
                try:
                    j = json.loads(o)
                    if "coordinates" in j and len(j["coordinates"]) >= 2:
                        lon, lat = j["coordinates"][:2]; return lat, lon
                except Exception:
                    return None, None
            return None, None
        latlon = df["the_geom"].apply(_coords)
        df["stop_lat"] = pd.to_numeric(latlon.apply(lambda t: t[0]), errors="coerce")
        df["stop_lon"] = pd.to_numeric(latlon.apply(lambda t: t[1]), errors="coerce")
        return df

    df["stop_lat"] = np.nan
    df["stop_lon"] = np.nan
    return df

def download_bus_with_retry(base_url: str, headers: dict, limit: int = 50000, max_retries: int = 5, backoff_base: float = 1.6):
    """Socrata API'den sayfalamalı indirme + 5xx/429 için retry/backoff."""
    rows, offset = [], 0
    while True:
        params = {"$limit": limit, "$offset": offset}
        attempt = 0
        while True:
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=60)
                if r.status_code in (429,) or 500 <= r.status_code < 600:
                    attempt += 1
                    if attempt > max_retries:
                        r.raise_for_status()
                    sleep_s = backoff_base ** attempt
                    print(f"⚠️ Geçici hata (status={r.status_code}) offset={offset} → {attempt}. deneme, {sleep_s:.1f}s bekleme...")
                    time.sleep(sleep_s); continue
                r.raise_for_status()
                data = r.json()
                chunk = pd.DataFrame(data)
                break
            except requests.HTTPError as e:
                print(f"❌ İndirme hatası (offset={offset}): {e}")
                return None
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"❌ Ağ/parse hatası (offset={offset}): {e}")
                    return None
                sleep_s = backoff_base ** attempt
                print(f"⚠️ Ağ/parse hatası (offset={offset}) → {attempt}. deneme, {sleep_s:.1f}s bekleme... ({e})")
                time.sleep(sleep_s)

        if chunk is None or chunk.empty:
            break
        if offset == 0:
            print("🔎 İlk chunk kolonları:", list(chunk.columns))
        rows.append(chunk)
        offset += len(chunk)
        print(f"  + {offset} kayıt indirildi...")
        if len(chunk) < limit:
            break

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)

# ========== yollar & ENV ==========
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# giriş/çıkışlar (kanonik + uyumluluk)
CRIME_INPUT        = os.path.join(BASE_DIR, os.getenv("CRIME_INPUT_NAME", "sf_crime_03.csv"))
CRIME_OUTPUT       = os.path.join(BASE_DIR, os.getenv("CRIME_OUTPUT_NAME", "sf_crime_04.csv"))

BUS_CANON_RAW      = os.path.join(BASE_DIR, os.getenv("BUS_CANON_RAW", "sf_bus_stops_with_geoid.csv"))  # canonical ham + GEOID
BUS_LEGACY_RAW_Y   = os.path.join(BASE_DIR, os.getenv("BUS_LEGACY_RAW_Y", "bus_y.csv"))                 # legacy ham
BUS_SUMMARY_NAME   = os.path.join(BASE_DIR, os.getenv("BUS_SUMMARY_NAME", "bus.csv"))                   # legacy özet/feature (GEOID-level)

# census geojson adayları
CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
    os.path.join(".",      "sf_census_blocks_with_population.geojson"),
]

# Socrata dataset ve token
RID    = os.getenv("BUS_DATASET_ID", "i28k-bkz6")  # SFMTA stops
BASE   = f"https://data.sfgov.org/resource/{RID}.json"
TOKEN  = os.getenv("SOCS_APP_TOKEN", "").strip()
HEADERS = {"Accept": "application/json"}
if TOKEN:
    HEADERS["X-App-Token"] = TOKEN

ALLOW_STUB = os.getenv("ALLOW_STUB_ON_API_FAIL", "1").strip().lower() not in ("0", "false")

# ========== 1) crime oku ==========
if not os.path.exists(CRIME_INPUT):
    raise FileNotFoundError(f"❌ Suç girdi dosyası yok: {CRIME_INPUT}")
crime = pd.read_csv(CRIME_INPUT, low_memory=False)
log_shape(crime, "CRIME (okundu)")
if "GEOID" not in crime.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime_geoids = pd.Series(crime["GEOID"].unique(), name="GEOID")
print(f"🧩 CRIME farklı GEOID sayısı: {crime_geoids.size}")

# ========== 2) bus indir/cache/stub ==========
print("🚌 Otobüs durakları Socrata'dan indiriliyor…")
bus = download_bus_with_retry(BASE, HEADERS, limit=50000, max_retries=5, backoff_base=1.7)

if bus is None:
    if os.path.exists(BUS_CANON_RAW):
        print("⚠️ API başarısız; mevcut cache kullanılacak:", os.path.abspath(BUS_CANON_RAW))
        bus = pd.read_csv(BUS_CANON_RAW, low_memory=False)
    elif os.path.exists(BUS_LEGACY_RAW_Y):
        print("⚠️ API başarısız; legacy cache kullanılacak:", os.path.abspath(BUS_LEGACY_RAW_Y))
        bus = pd.read_csv(BUS_LEGACY_RAW_Y, low_memory=False)
    elif ALLOW_STUB:
        print("⚠️ API ve yerel cache yok → STUB (0 durak, NaN mesafe).")
        bus = pd.DataFrame(columns=["stop_lat", "stop_lon"])
    else:
        raise SystemExit("⚠️ Otobüs durakları alınamadı; cache de yok.")

# lat/lon çıkar
bus = extract_lat_lon(bus)
bus = bus.dropna(subset=["stop_lat", "stop_lon"]).copy()
log_shape(bus, "BUS (lat/lon sonrası)")

# stop_id normalize (varsa)
for cand in ["stop_id", "stopid", "stop", "id"]:
    if cand in bus.columns:
        bus.rename(columns={cand: "stop_id"}, inplace=True)
        break

# ========== 3) bloklar & GEOID ata ==========
census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
blocks_ok = True
if census_path is None:
    print("⚠️ Nüfus blokları GeoJSON bulunamadı. GEOID eşleme/mesafe → stub.")
    blocks_ok = False

if blocks_ok:
    gdf_blocks = gpd.read_file(census_path)
    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True, allow_override=True)
    else:
        try:
            epsg = gdf_blocks.crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            gdf_blocks = gdf_blocks.to_crs(epsg=4326)

    gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next(
        (c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None
    )
    if not gcol:
        print("⚠️ Block dosyasında GEOID benzeri sütun yok. Stub'a düşülecek.")
        blocks_ok = False
    else:
        gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)
else:
    gdf_blocks = None

if blocks_ok and not bus.empty:
    gdf_bus = gpd.GeoDataFrame(
        bus, geometry=gpd.points_from_xy(bus["stop_lon"], bus["stop_lat"]), crs="EPSG:4326"
    )
    try:
        gdf_bus = gpd.sjoin(gdf_bus, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    except Exception as e:
        print(f"⚠️ sjoin(within) başarısız ({e}). sjoin_nearest(max_distance=5 m) deneniyor…")
        gdf_bus = gpd.sjoin_nearest(gdf_bus, gdf_blocks[["GEOID", "geometry"]], how="left", max_distance=5)
    gdf_bus = gdf_bus.drop(columns=["index_right"], errors="ignore")
    gdf_bus["GEOID"] = normalize_geoid(gdf_bus["GEOID"], DEFAULT_GEOID_LEN)
    bus_geo = pd.DataFrame(gdf_bus.drop(columns=["geometry"], errors="ignore")).copy()
    log_shape(bus_geo, "BUS⨯BLOCKS (GEOID atanmış)")
else:
    bus_geo = bus.copy()
    bus_geo["GEOID"] = pd.NA

# ========== 4) ham (canonical + legacy) KAYDET ==========
safe_save_csv(bus_geo, BUS_CANON_RAW)   # canonical
try:
    safe_save_csv(bus_geo, BUS_LEGACY_RAW_Y)  # legacy uyumluluk
    print(f"✅ BUS ham (canonical): {BUS_CANON_RAW}")
    print(f"↪️ Legacy kopya (bus_y.csv): {BUS_LEGACY_RAW_Y}")
except Exception as e:
    print(f"⚠️ Legacy bus_y.csv yazılamadı: {e}")

# ========== 5) GEOID-özet (count & en yakın mesafe) ==========
if blocks_ok and not bus_geo["GEOID"].isna().all():
    # count
    bus_count = (
        bus_geo.dropna(subset=["GEOID"])
               .groupby("GEOID", as_index=False)
               .agg(bus_stop_count=("stop_lat", "size"))
    )
    # distance (m) — blok centroid → en yakın durak
    gdf_blocks_xy = gdf_blocks[["GEOID", "geometry"]].copy().to_crs(3857)
    gdf_blocks_xy["cx"] = gdf_blocks_xy.geometry.centroid.x
    gdf_blocks_xy["cy"] = gdf_blocks_xy.geometry.centroid.y

    gdf_bus_xy = gpd.GeoDataFrame(
        bus_geo.dropna(subset=["stop_lat", "stop_lon"])[["stop_lat", "stop_lon"]].copy(),
        geometry=gpd.points_from_xy(bus_geo.dropna(subset=["stop_lat", "stop_lon"])["stop_lon"],
                                    bus_geo.dropna(subset=["stop_lat", "stop_lon"])["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(3857)

    if len(gdf_bus_xy) == 0:
        bus_dist = gdf_blocks_xy[["GEOID"]].copy()
        bus_dist["distance_to_bus"] = np.nan
    else:
        bus_coords = np.vstack([gdf_bus_xy.geometry.x.values, gdf_bus_xy.geometry.y.values]).T
        tree = cKDTree(bus_coords)
        centroids = np.vstack([gdf_blocks_xy["cx"].values, gdf_blocks_xy["cy"].values]).T
        distances, _ = tree.query(centroids, k=1)
        bus_dist = gdf_blocks_xy[["GEOID"]].copy()
        bus_dist["distance_to_bus"] = distances.astype(float)
else:
    bus_count = pd.DataFrame(columns=["GEOID", "bus_stop_count"])
    bus_dist = pd.DataFrame({"GEOID": crime_geoids, "distance_to_bus": np.nan})

# özet tablo
bus_feat = pd.merge(bus_dist, bus_count, on="GEOID", how="left")
bus_feat["bus_stop_count"] = bus_feat["bus_stop_count"].fillna(0).astype(int)
bus_feat["GEOID"] = normalize_geoid(bus_feat["GEOID"], DEFAULT_GEOID_LEN)

# sadece crime’da olan GEOID’leri tut
bus_feat = bus_feat.merge(crime_geoids.to_frame(), on="GEOID", how="right")

# binleme alanları (opsiyonel, görselleme/feature için faydalı)
d = bus_feat["distance_to_bus"].replace([np.inf, -np.inf], np.nan).dropna()
if len(d) >= 2 and d.max() > d.min():
    n_bins = freedman_diaconis_bin_count(d.to_numpy(), max_bins=10)
    _, dist_edges = pd.qcut(d, q=n_bins, retbins=True, duplicates="drop")
    dist_labels = [f"{int(dist_edges[i])}–{int(dist_edges[i+1])}m" for i in range(len(dist_edges) - 1)]
    bus_feat["distance_to_bus_range"] = pd.cut(
        bus_feat["distance_to_bus"], bins=dist_edges, labels=dist_labels, include_lowest=True
    )
else:
    bus_feat["distance_to_bus_range"] = pd.Series(["0–0m"] * len(bus_feat))

cnt = bus_feat["bus_stop_count"].fillna(0)
if cnt.nunique() > 1:
    n_c_bins = freedman_diaconis_bin_count(cnt.to_numpy(), max_bins=8)
    _, cnt_edges = pd.qcut(cnt, q=n_c_bins, retbins=True, duplicates="drop")
    cnt_labels = [f"{int(cnt_edges[i])}–{int(cnt_edges[i+1])}" for i in range(len(cnt_edges) - 1)]
    bus_feat["bus_stop_count_range"] = pd.cut(cnt, bins=cnt_edges, labels=cnt_labels, include_lowest=True)
else:
    bus_feat["bus_stop_count_range"] = pd.Series([f"{int(cnt.min())}–{int(cnt.max())}"] * len(cnt))

bus_feat = bus_feat.sort_values("GEOID").drop_duplicates(subset="GEOID", keep="first")
assert bus_feat["GEOID"].is_unique, "BUS: GEOID hâlâ tekil değil!"
log_shape(bus_feat, "BUS özellikleri (GEOID düzeyi)")

# ========== 6) özet dosyayı da kaydet (legacy: bus.csv) ==========
safe_save_csv(bus_feat, BUS_SUMMARY_NAME)
print(f"✅ BUS özet (GEOID-level) yazıldı → {BUS_SUMMARY_NAME}")

# ========== 7) crime ile merge ==========
_before = crime.shape
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
bus_feat["GEOID"] = normalize_geoid(bus_feat["GEOID"], DEFAULT_GEOID_LEN)

crime = crime.merge(bus_feat, on="GEOID", how="left", validate="many_to_one")
crime["bus_stop_count"] = crime["bus_stop_count"].fillna(0).astype(int)

log_delta(_before, crime.shape, "CRIME ⨯ BUS (GEOID enrich)")
log_shape(crime, "CRIME (bus enrich sonrası)")

# ========== 8) kayıtlar ==========
safe_save_csv(crime, CRIME_OUTPUT)
print(f"📁 CRIME çıktı → {CRIME_OUTPUT}")

try:
    print("sf_crime_04.csv — ilk 5 satır")
    print(crime.head(5).to_string(index=False))
except Exception:
    pass
