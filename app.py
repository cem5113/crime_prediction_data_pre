# app.py
from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
import requests
import re
import os, json, subprocess, sys
from pathlib import Path
import io, zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

# --- Forensic rapor yardımcı (varsa import et, yoksa stub kullan) ---
try:
    from scripts.forensic_report import build_forensic_report
except Exception:
    def build_forensic_report(**kwargs):
        return None

st.set_page_config(page_title="Veri Güncelleme (Parquet)", layout="wide")

# =========================
# Parquet-first yardımcılar
# =========================

def _ensure_pyarrow():
    try:
        import pyarrow  # noqa
    except Exception as e:
        st.error("Parquet I/O için 'pyarrow' gerekli. Lütfen 'pip install pyarrow' kurun.")
        st.stop()

_ensure_pyarrow()

def read_df(path: Union[str, Path]) -> pd.DataFrame:
    """Parquet öncelikli oku; CSV ise otomatik oku."""
    p = Path(path)
    if p.suffix.lower() == ".parquet" and p.exists():
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv" and p.exists():
        return pd.read_csv(p, low_memory=False)
    # uygun parquet varsa onu kullan
    pq = p.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    # CSV varsa fallback + dönüştür
    csvp = p.with_suffix(".csv")
    if csvp.exists():
        df = pd.read_csv(csvp, low_memory=False)
        try:
            df.to_parquet(pq, index=False)
        except Exception:
            pass
        return df
    raise FileNotFoundError(f"Bulunamadı: {p} (veya {pq} / {csvp})")

def write_df(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Parquet olarak yaz; dosya uzantısı .csv verilse bile .parquet üretir."""
    p = Path(path)
    if p.suffix.lower() != ".parquet":
        p = p.with_suffix(".parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p

def csv_to_parquet_if_needed(src_csv: Union[str, Path], dst_parquet: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """CSV'yi parquet'e çevirir; dst verilmezse aynı ada .parquet yazar."""
    try:
        src = Path(src_csv)
        if not src.exists():
            return None
        dst = Path(dst_parquet) if dst_parquet else src.with_suffix(".parquet")
        df = pd.read_csv(src, low_memory=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        return dst
    except Exception:
        return None

# =========================
# Config / ENV
# =========================

PIPELINE = [
    {"name": "update_crime.py",      "alts": ["build_crime_grid.py", "crime_grid_build.py"]},
    {"name": "update_911.py",        "alts": ["enrich_911.py"]},
    {"name": "update_311.py",        "alts": ["enrich_311.py"]},
    {"name": "update_population.py", "alts": ["enrich_population.py"]},
    {"name": "update_bus.py",        "alts": ["enrich_bus.py"]},
    {"name": "update_train.py",      "alts": ["enrich_train.py"]},
    {"name": "update_poi.py",        "alts": ["pipeline_make_sf_crime_06.py", "app_poi_to_06.py", "enrich_poi.py"]},
    {"name": "update_police_gov.py", "alts": ["enrich_police_gov_06_to_07.py", "enrich_police_gov.py", "enrich_police.py"]},
    {"name": "update_weather.py",    "alts": ["enrich_weather.py"]},
]

def _load_ml_deps():
    try:
        import shap
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.inspection import PartialDependenceDisplay
        from lightgbm import LGBMClassifier, LGBMRegressor
        from lime.lime_tabular import LimeTabularExplainer
        return {
            "np": np,
            "shap": shap,
            "TimeSeriesSplit": TimeSeriesSplit,
            "roc_auc_score": roc_auc_score,
            "brier_score_loss": brier_score_loss,
            "mean_absolute_error": mean_absolute_error,
            "CalibratedClassifierCV": CalibratedClassifierCV,
            "PartialDependenceDisplay": PartialDependenceDisplay,
            "LGBMClassifier": LGBMClassifier,
            "LGBMRegressor": LGBMRegressor,
            "LimeTabularExplainer": LimeTabularExplainer,
        }
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "paket")
        st.error(f"🧱 Gerekli paket eksik: **{missing}**. Sol menüde '0) Gereklilikleri yükle' ile kurup Rerun yapın.")
        st.stop()

def pick_url(key: str, default: str) -> str:
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

CRIME_CSV_LATEST = pick_url("CRIME_CSV_URL", "https://github.com/cem5113/crime_prediction_data_pre/releases/latest/download/sf_crime_y.csv")
RAW_911_URL = pick_url("RAW_911_URL", "https://github.com/cem5113/crime_prediction_data_pre/releases/download/v1.0.1/sf_911_last_5_year_y.csv")
SF311_URL = pick_url("SF311_URL", "https://github.com/cem5113/crime_prediction_data_pre/releases/download/v1.0.2/sf_311_last_5_years_y.csv")

DEFAULT_POP_CSV = str((Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data_pre")) / "sf_population.csv").resolve())
POPULATION_PATH = pick_url("POPULATION_PATH", DEFAULT_POP_CSV)
if re.match(r"^https?://", str(POPULATION_PATH), flags=re.I):
    POPULATION_PATH = DEFAULT_POP_CSV
os.environ["POPULATION_PATH"] = str(POPULATION_PATH)

SF911_API_URL       = pick_url("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF911_AGENCY_FILTER = pick_url("SF911_AGENCY_FILTER", "agency like '%Police%'")
SF911_API_TOKEN     = pick_url("SF911_API_TOKEN", "")

os.environ["CRIME_CSV_URL"] = CRIME_CSV_LATEST
os.environ["RAW_911_URL"]   = RAW_911_URL
os.environ["SF311_URL"]     = SF311_URL
os.environ["GEOID_LEN"]     = os.environ.get("GEOID_LEN", "11")
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .str[:L]
         .str.zfill(L)
    )

os.environ["SF911_API_URL"] = SF911_API_URL
os.environ["SF911_AGENCY_FILTER"] = SF911_AGENCY_FILTER
if SF911_API_TOKEN:
    os.environ["SF911_API_TOKEN"] = SF911_API_TOKEN
SOCS_APP_TOKEN = st.secrets.get("SOCS_APP_TOKEN", os.environ.get("SOCS_APP_TOKEN", ""))
if SOCS_APP_TOKEN:
    os.environ["SOCS_APP_TOKEN"] = SOCS_APP_TOKEN

os.environ["ACS_YEAR"] = st.secrets.get("ACS_YEAR", os.environ.get("ACS_YEAR", "LATEST"))
os.environ["DEMOG_WHITELIST"] = st.secrets.get("DEMOG_WHITELIST", os.environ.get("DEMOG_WHITELIST", ""))

GITHUB_REPO = os.environ.get("GITHUB_REPO", "cem5113/crime_prediction_data_pre")
GITHUB_WORKFLOW = os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")

def _gh_headers():
    token = st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN gerekli (Streamlit secrets veya env).")
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

def fetch_file_from_latest_artifact(pick_names: list[str], artifact_name="sf-crime-pipeline-output") -> bytes | None:
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
    runs = requests.get(runs_url, headers=_gh_headers(), timeout=30).json()
    run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
    for rid in run_ids:
        arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
        arts = requests.get(arts_url, headers=_gh_headers(), timeout=30).json().get("artifacts", [])
        for a in arts:
            if a.get("name") == artifact_name and not a.get("expired", False):
                dl = requests.get(a["archive_download_url"], headers=_gh_headers(), timeout=60)
                zf = zipfile.ZipFile(io.BytesIO(dl.content))
                names = zf.namelist()
                for pick in pick_names:
                    for c in (pick, f"crime_prediction_data_pre/{pick}"):
                        if c in names:
                            return zf.read(c)
                for n in names:
                    if any(n.endswith(p) for p in pick_names):
                        return zf.read(n)
    return None
    
def dispatch_workflow(wf_selector: str, persist="artifact", force=True, top_k="50", ref="main"):
    import json as _json, base64
    base = f"https://api.github.com/repos/{GITHUB_REPO}"
    headers = _gh_headers()

    # 1) Workflow listesinde eşle: path (endswith), isim (name) veya id
    lst = requests.get(f"{base}/actions/workflows", headers=headers, timeout=20).json()
    wf_id = None; wf_path = None; wf_state = None
    for w in lst.get("workflows", []):
        if (
            str(w.get("id")) == str(wf_selector)
            or w.get("name","") == wf_selector
            or w.get("path","").endswith(f"/{wf_selector}")
        ):
            wf_id = w["id"]; wf_path = w.get("path"); wf_state = w.get("state"); break

    if not wf_id:
        return {"ok": False, "status": 404, "text": f"Workflow bulunamadı: {wf_selector}"}

    # 2) İçeriği kontrol et: gerçekten workflow_dispatch var mı?
    yml = requests.get(f"{base}/contents/{wf_path}", headers=headers, timeout=20).json()
    content = ""
    try:
        if "content" in yml:
            content = base64.b64decode(yml["content"]).decode("utf-8","ignore")
    except Exception:
        pass
    has_dispatch = "workflow_dispatch" in content

    # 3) Dispatch (ref = seçilen branch)
    payload = {"ref": ref, "inputs": {"persist": persist, "force": "true" if force else "false", "top_k": top_k}}
    resp = requests.post(f"{base}/actions/workflows/{wf_id}/dispatches", headers=headers, json=payload, timeout=30)
    return {
        "ok": resp.status_code in (204, 201),
        "status": resp.status_code,
        "text": resp.text,
        "wf_id": wf_id,
        "state": wf_state,
        "has_dispatch": has_dispatch,
        "path": wf_path,
        "ref": ref,
    }
    
def _get_last_run_by_workflow():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/runs?per_page=1"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code != 200:
        return None, r.status_code, r.text
    arr = r.json().get("workflow_runs", [])
    return (arr[0] if arr else None), 200, ""

def _render_last_run_status(container):
    if not (st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")):
        container.info("GH_TOKEN yok; GitHub durumunu okuyamıyorum.")
        return
    try:
        run, code, msg = _get_last_run_by_workflow()
        if not run:
            container.info("Bu workflow için run bulunamadı.")
            return
        status = run.get("status")
        concl  = run.get("conclusion") or "-"
        started = run.get("run_started_at")
        html_url = run.get("html_url")
        container.markdown(f"**Son koşum:** `{status}` / `{concl}` · başlama: `{started}` · [GitHub’da aç]({html_url})")
    except Exception as e:
        container.warning(f"Durum okunamadı: {e}")

def fetch_latest_artifact_df() -> Optional[pd.DataFrame]:
    try:
        runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
        runs = requests.get(runs_url, headers=_gh_headers(), timeout=30).json()
        run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
        for rid in run_ids:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=_gh_headers(), timeout=30).json().get("artifacts", [])
            for a in arts:
                if a.get("name") == "sf-crime-pipeline-output" and not a.get("expired", False):
                    dl = requests.get(a["archive_download_url"], headers=_gh_headers(), timeout=60)
                    zf = zipfile.ZipFile(io.BytesIO(dl.content))
                    for pick in ("crime_prediction_data_pre/sf_crime_08.csv", "sf_crime_08.csv"):
                        if pick in zf.namelist():
                            with zf.open(pick) as f:
                                df = pd.read_csv(f, low_memory=False)
                                if "date" in df.columns:
                                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                                elif "datetime" in df.columns:
                                    df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
                                return df
        return None
    except Exception as e:
        st.warning(f"Artifact indirilemedi: {e}")
        return None

# ================
# Yol kurulumları
# ================
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
DATA_DIR = ROOT / "crime_prediction_data_pre"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIRS = [SCRIPTS_DIR, ROOT]

def _mask_token(u: str) -> str:
    try:
        return re.sub(r'(\$\$app_token=)[^&]+', r'\1•••', str(u))
    except:
        return str(u)

# ====================================
# İndirilebilirler (Parquet kaydetme)
# ====================================
DOWNLOADS = {
    "Suç Taban (latest, CSV → Parquet)": {
        "url": CRIME_CSV_LATEST,
        "csv_path": str(DATA_DIR / "sf_crime_y.csv"),
        "parquet_path": str(DATA_DIR / "sf_crime_y.parquet"),
    },
    "Tahmin Grid (CSV → Parquet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_crime_grid_full_labeled.csv",
        "csv_path": str(DATA_DIR / "sf_crime_grid_full_labeled.csv"),
        "parquet_path": str(DATA_DIR / "sf_crime_grid_full_labeled.parquet"),
        "allow_artifact": True,
        "artifact_picks": ["sf_crime_grid_full_labeled.csv"],
    },
    "911 Çağrıları (CSV → Parquet)": {
        "url": RAW_911_URL,
        "csv_path": str(DATA_DIR / "sf_911_last_5_year_y.csv"),
        "parquet_path": str(DATA_DIR / "sf_911_last_5_year_y.parquet"),
    },
    "311 Çağrıları (CSV → Parquet)": {
        "url": SF311_URL,
        "csv_path": str(DATA_DIR / "sf_311_last_5_years_y.csv"),
        "parquet_path": str(DATA_DIR / "sf_311_last_5_years_y.parquet"),
    },
    "Otobüs Durakları (CSV → Parquet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_bus_stops_with_geoid.csv",
        "csv_path": str(DATA_DIR / "sf_bus_stops_with_geoid.csv"),
        "parquet_path": str(DATA_DIR / "sf_bus_stops_with_geoid.parquet"),
    },
    "Tren Durakları (CSV → Parquet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_train_stops_with_geoid.csv",
        "csv_path": str(DATA_DIR / "sf_train_stops_with_geoid.csv"),
        "parquet_path": str(DATA_DIR / "sf_train_stops_with_geoid.parquet"),
    },
    # JSON dosyaları JSON olarak tutulur
    "POI GeoJSON (JSON)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_pois.geojson",
        "json_path": str(DATA_DIR / "sf_pois.geojson"),
        "is_json": True,
    },
    "Nüfus Verisi (Yerel CSV → Parquet)": {
        "url": "",
        "csv_path": str(DATA_DIR / "sf_population.csv"),
        "parquet_path": str(DATA_DIR / "sf_population.parquet"),
        "local_src": str(POPULATION_PATH),
        "is_local_csv": True,
    },
    "POI Risk Skorları (JSON)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/risky_pois_dynamic.json",
        "json_path": str(DATA_DIR / "risky_pois_dynamic.json"),
        "is_json": True,
    },
    "Polis İstasyonları (CSV → Parquet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_police_stations.csv",
        "csv_path": str(DATA_DIR / "sf_police_stations.csv"),
        "parquet_path": str(DATA_DIR / "sf_police_stations.parquet"),
    },
    "Devlet Binaları (CSV → Parquet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_government_buildings.csv",
        "csv_path": str(DATA_DIR / "sf_government_buildings.csv"),
        "parquet_path": str(DATA_DIR / "sf_government_buildings.parquet"),
    },
    "Hava Durumu (CSV → Parquet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data_pre/main/sf_weather_5years_y.csv",
        "csv_path": str(DATA_DIR / "sf_weather_5years_y.csv"),
        "parquet_path": str(DATA_DIR / "sf_weather_5years_y.parquet"),
    },
}

def _human_bytes(n: int) -> str:
    if n is None: return "-"
    step = 1024.0
    for u in ["B","KB","MB","GB","TB"]:
        if n < step: return f"{n:.0f} {u}" if u=="B" else f"{n:.1f} {u}"
        n /= step
    return f"{n:.1f} PB"

def _fmt_dt(ts: Optional[float]) -> str:
    if ts is None: return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _age_str(ts: Optional[float]) -> str:
    if ts is None: return "-"
    delta = datetime.now().timestamp() - ts
    if delta < 60:   return f"{int(delta)} sn"
    if delta < 3600: return f"{int(delta//60)} dk"
    if delta < 86400:return f"{int(delta//3600)} sa"
    return f"{int(delta//86400)} g"

def list_files_sorted(
    include: Optional[List[Union[str, Path]]] = None,
    base_dir: Optional[Path] = None,
    pattern: str = "*.parquet",
    ascending: bool = True,
    include_missing: bool = True,
) -> pd.DataFrame:
    bdir = base_dir or DATA_DIR
    rows: List[Dict[str, Any]] = []
    if include is None:
        include = []
        # Tüm Parquet adaylarını dolaş
        for v in DOWNLOADS.values():
            pq = v.get("parquet_path") or v.get("json_path") or v.get("csv_path")
            if pq: include.append(str(pq))
        include += [str(bdir / f"sf_crime_{i:02d}.parquet") for i in range(1, 10)]
        include += [str(bdir / "sf_crime_y.parquet"),
                    str(bdir / "sf_crime_grid_full_labeled.parquet"),
                    str(bdir / "sf_crime_08.parquet"),
                    str(bdir / "sf_crime_09.parquet"),
                    str(bdir / "sf_crime_09_with_preds.parquet")]
        for p in bdir.glob(pattern):
            include.append(str(p))

    seen = set()
    for x in include:
        p = Path(x)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen: continue
        seen.add(key)
        exists = p.exists()
        try:
            st_ = p.stat() if exists else None
            mtime = st_.st_mtime if st_ else None
            size  = st_.st_size  if st_ else None
        except Exception:
            mtime, size = None, None
        if exists or include_missing:
            rows.append({
                "file": p.name,
                "path": str(p),
                "exists": bool(exists),
                "size": _human_bytes(size),
                "modified": _fmt_dt(mtime),
                "age": _age_str(mtime),
                "_mtime": mtime,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_mtime", ascending=ascending, na_position="last").drop(columns=["_mtime"])
    return df

# ============================
# 0) (Opsiyonel) requirements
# ============================
st.markdown("### 0) (Opsiyonel) Gereklilikleri yükle")
if st.button("📦 requirements.txt yükle"):
    try:
        req = ROOT / "requirements.txt"
        if req.exists():
            out = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            st.code(out.stdout or "")
            if out.returncode == 0:
                st.success("✅ Gereklilikler yüklendi.")
            else:
                st.error("❌ Kurulumda hata!")
                st.code(out.stderr or "")
        else:
            st.warning("⚠️ requirements.txt bulunamadı.")
    except Exception as e:
        st.error(f"Kurulum çağrısı başarısız: {e}")

# ============================
# İndir & Parquet’e dönüştür
# ============================
def download_and_preview_parquet(name, info):
    st.markdown(f"### 🔹 {name}")
    url = info.get("url")
    csv_path = info.get("csv_path")
    parquet_path = info.get("parquet_path")
    json_path = info.get("json_path")
    is_json = info.get("is_json", False)
    allow_artifact_fallback = info.get("allow_artifact", False)
    artifact_picks = info.get("artifact_picks")

    if is_json:
        if not url:
            st.warning("URL boş, atlanıyor.")
            return
        st.caption(f"URL: {_mask_token(url)}")
        try:
            r = requests.get(url, timeout=60); r.raise_for_status()
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(json_path).write_text(r.text, encoding="utf-8")
            st.success("✅ JSON indirildi.")
            try:
                data = json.loads(Path(json_path).read_text(encoding="utf-8"))
                if isinstance(data, dict): st.json(data)
                elif isinstance(data, list): st.json(data[:3])
                else: st.code(str(data)[:1000])
            except Exception as e:
                st.code(Path(json_path).read_text(encoding="utf-8")[:2000])
        except Exception as e:
            st.error(f"❌ JSON indirilemedi: {e}")
        return

    # CSV indirme (artifact fallback) ve parquet’e çevirme
    ok = False
    if url:
        st.caption(f"URL: {_mask_token(url)}")
        try:
            r = requests.get(url, timeout=60); r.raise_for_status()
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "wb") as f:
                f.write(r.content)
            ok = True
        except Exception as e:
            st.warning(f"Raw indirme başarısız: {e}")
    if (not ok) and allow_artifact_fallback:
        try:
            blob = fetch_file_from_latest_artifact(artifact_picks or [os.path.basename(csv_path)])
            if blob:
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                with open(csv_path, "wb") as f:
                    f.write(blob)
                ok = True
                st.info("Dosya artifact'tan alındı.")
        except Exception as e:
            st.warning(f"Artifact fallback başarısız: {e}")

    # Yerel CSV kopyası (nüfus gibi)
    if info.get("is_local_csv"):
        src = Path(info.get("local_src"))
        if src.exists():
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                Path(csv_path).write_bytes(src.read_bytes())
                ok = True
                st.info("Yerel CSV kopyalandı.")
            except Exception as e:
                st.warning(f"Yerel CSV kopyalanamadı: {e}")

    if not ok and (not Path(csv_path).exists()):
        st.error("❌ İndirilemedi.")
        return

    # CSV -> Parquet
    pq = csv_to_parquet_if_needed(csv_path, parquet_path)
    if pq:
        st.success(f"✅ Parquet’e dönüştürüldü: {pq.name}")
        try:
            head = pd.read_parquet(pq).head(3)
            st.dataframe(head)
            st.caption(f"📌 Sütunlar: {list(head.columns)}")
        except Exception as e:
            st.info("Önizleme başarısız; Parquet yazıldı.")
            st.code(f"Önizleme hatası: {e}")
    else:
        st.warning("Parquet’e dönüştürülemedi; CSV elde var.")
        try:
            head = pd.read_csv(csv_path, nrows=3)
            st.dataframe(head)
            st.caption(f"📌 Sütunlar: {list(head.columns)}")
        except Exception as e:
            st.info("CSV önizleme de başarısız.")

st.markdown("### 1) (Opsiyonel) Verileri indir → Parquet’e dönüştür → Önizle")
if st.button("📥 İndir / Dönüştür / Önizle"):
    for name, info in DOWNLOADS.items():
        download_and_preview_parquet(name, info)
    st.success("✅ Bitti.")

st.markdown("### 1.5) Dosyaları tarihe göre sırala (Parquet)")
colA, colB, colC = st.columns([1,1,2])
with colA:
    order = st.radio("Sıralama", ["Eski → Yeni", "Yeni → Eski"], horizontal=True, index=0)
with colB:
    show_missing = st.checkbox("Eksikleri de göster", value=True)
with colC:
    patt = st.text_input("Desen (glob)", "*.parquet", help="Örn: sf_crime_*.parquet", key="glob_list")
asc = (order == "Eski → Yeni")
df_files = list_files_sorted(pattern=patt, ascending=asc, include_missing=show_missing)
st.dataframe(df_files if not df_files.empty else pd.DataFrame([{"info":"Eşleşen dosya yok."}]))

with st.expander("🔎 Tanı: Etkin URL/ENV değerleri"):
    st.write("CRIME_CSV_URL (env):", os.environ.get("CRIME_CSV_URL"))
    st.write("RAW_911_URL (env):", os.environ.get("RAW_911_URL"))
    st.write("SF311_URL (env):", os.environ.get("SF311_URL"))

if st.button("♻️ Streamlit cache temizle"):
    try:
        st.cache_data.clear()
        st.success("Cache temizlendi.")
    except Exception as e:
        st.warning(f"Cache temizlenemedi: {e}")

# ======================
# Script bul/çalıştır
# ======================
def ensure_script(local_name: str) -> Optional[Path]:
    for d in SEARCH_DIRS:
        p = d / local_name
        if p.exists():
            return p
    return None

def resolve_script(entry: dict) -> Optional[Path]:
    p = ensure_script(entry["name"])
    if p:
        return p
    for alt in entry.get("alts", []):
        pp = ensure_script(alt)
        if pp:
            target = SCRIPTS_DIR / entry["name"]
            try:
                target.write_text(Path(pp).read_text(encoding="utf-8"), encoding="utf-8")
                st.info(f"🔁 {alt} → {entry['name']} olarak kopyalandı.")
                return target
            except Exception:
                return pp
    return None

def run_script(path: Path) -> bool:
    st.write(f"▶️ {path.name} çalıştırılıyor…")
    placeholder = st.empty()
    lines = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(path)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip())
                lines = lines[-400:]
                placeholder.code("\n".join(lines))
        rc = proc.wait()
        if rc == 0:
            st.success(f"✅ {path.name} tamamlandı")
            return True
        else:
            st.error(f"❌ {path.name} hata verdi (exit={rc})")
            return False
    except Exception as e:
        st.error(f"🚨 {path.name} çağrılamadı: {e}")
        return False

st.markdown("### 2) Güncelleme ve Zenginleştirme (01 → 09)")
if st.button("⚙️ Güncelleme ve Zenginleştirme (01 → 09)"):
    with st.spinner("⏳ Scriptler çalıştırılıyor..."):
        all_ok = True
        for entry in PIPELINE:
            sp = resolve_script(entry)
            if not sp:
                st.warning(f"⏭️ {entry['name']} bulunamadı/indirilemedi, atlanıyor.")
                all_ok = False
                continue
            ok = run_script(sp)
            all_ok = all_ok and ok
    if all_ok:
        st.success("🎉 Pipeline bitti: Tüm adımlar başarıyla tamamlandı.")
    else:
        st.warning("ℹ️ Pipeline tamamlandı; eksik/hatalı adımlar var. Logları kontrol edin.")

# ===================================
# sf_crime_08 yükle (Parquet-first)
# ===================================
def load_sf_crime_08(local_path: Path) -> Optional[pd.DataFrame]:
    """Önce Parquet, sonra CSV; yoksa artifact CSV → Parquet."""
    def _normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "datetime" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
        return df

    df: Optional[pd.DataFrame] = None
    pq = local_path.with_suffix(".parquet")
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
            df = _normalize_date_cols(df)
        except Exception as e:
            st.warning(f"Yerel Parquet okunamadı: {e}")

    if df is None and local_path.with_suffix(".csv").exists():
        try:
            df = pd.read_csv(local_path.with_suffix(".csv"), low_memory=False)
            df = _normalize_date_cols(df)
            try:
                df.to_parquet(pq, index=False)
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Yerel CSV okunamadı: {e}")

    if df is None:
        # artifact’tan CSV → Parquet
        df = fetch_latest_artifact_df()
        if df is not None:
            df = _normalize_date_cols(df)
            try:
                pd.DataFrame(df).to_parquet(pq, index=False)
                st.info(f"Artifact verisi Parquet’e yazıldı: {pq.name}")
            except Exception:
                pass
    return df

# ==========================================
# Rare grouping + zenginleştir + Parquet yaz
# ==========================================
def _group_rare_labels(
    df: pd.DataFrame,
    col: str,
    min_prop: Optional[float] = None,
    min_count: Optional[int] = None,
    other_label: str = "Other",
    out_stats_path: Optional[Path] = None,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    s = df[col].astype(str).str.strip()
    total = len(s)
    vc = s.value_counts(dropna=False)
    env_prop = os.environ.get("RARE_MIN_PROP")
    env_count = os.environ.get("RARE_MIN_COUNT")
    if min_prop is None and env_prop:
        try: min_prop = float(env_prop)
        except: pass
    if min_count is None and env_count:
        try: min_count = int(env_count)
        except: pass
    if min_prop is None and min_count is None:
        min_prop, min_count = 0.01, 200
    rare_mask = pd.Series(False, index=vc.index)
    if min_prop is not None:
        rare_mask |= (vc / max(total, 1)) < float(min_prop)
    if min_count is not None:
        rare_mask |= vc < int(min_count)
    rare_values = set(vc[rare_mask].index)
    grouped = s.where(~s.isin(rare_values), other_label)
    if out_stats_path is not None:
        out_stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df = pd.DataFrame({
            col: vc.index,
            "count": vc.values,
            "prop": vc.values / max(total, 1),
            "is_rare": vc.index.map(lambda v: v in rare_values)
        })
        try:
            stats_df.to_parquet(out_stats_path.with_suffix(".parquet"), index=False)
        except Exception:
            try: stats_df.to_csv(out_stats_path.with_suffix(".csv"), index=False)
            except Exception: pass
    return grouped

def clean_and_save_crime_09(input_obj: Union[str, Path, pd.DataFrame]="sf_crime_08.parquet",
                             output_path: Union[str, Path]="sf_crime_09.parquet") -> pd.DataFrame:
    # input: DataFrame veya dosya yolu (Parquet/CSV)
    if isinstance(input_obj, pd.DataFrame):
        df = input_obj.copy()
    else:
        df = read_df(input_obj)

    if "GEOID" in df.columns:
        L = int(os.environ.get("GEOID_LEN", "11"))
        df["GEOID"] = (
            df["GEOID"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(L)
        )

    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.title()

    # Rare grouping
    try:
        out_dir = Path(output_path).parent if isinstance(output_path, (str, Path)) else Path(".")
        if "category" in df.columns:
            df["category_grouped"] = _group_rare_labels(
                df, "category", None, None, "Other", out_stats_path=out_dir / "rare_stats_category"
            )
        if "subcategory" in df.columns:
            df["subcategory"] = df["subcategory"].astype(str).str.strip().str.title()
            df["subcategory_grouped"] = _group_rare_labels(
                df, "subcategory", None, None, "Other", out_stats_path=out_dir / "rare_stats_subcategory"
            )
        try: st.caption("🔎 Rare grouping uygulandı (category/subcategory). İstatistikler kaydedildi.")
        except: pass
    except Exception as _e:
        try: st.warning(f"Rare grouping atlandı: {str(_e)}")
        except: print(f"Rare grouping atlandı: {_e}")

    # Dönüştürücüler
    def to_int(df, col, default=0):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).round().astype("Int64")

    def to_float(df, col, default=0.0):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)

    # 1) Sayaçlar
    for c in [
        "crime_count", "911_request_count_hour_range", "911_request_count_daily(before_24_hours)",
        "311_request_count", "bus_stop_count", "train_stop_count", "poi_total_count",
    ]:
        to_int(df, c, default=0)

    # 1-b) Risk skoru
    to_float(df, "poi_risk_score", default=0.0)

    # 2) Binary kolonlar
    def to_binary(df, col):
        if col in df.columns:
            m = {"true":1,"t":1,"yes":1,"y":1,"1":1,"evet":1, "false":0,"f":0,"no":0,"n":0,"0":0,"hayır":0,"hayir":0}
            s = df[col].replace({True:1, False:0})
            s = s.astype(str).str.strip().str.lower().map(m)
            df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")
    for c in ["is_near_police", "is_near_government"]:
        to_binary(df, c)

    # 3) Mesafeler
    for c in ["distance_to_bus", "distance_to_train", "distance_to_police", "distance_to_government_building"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(9999.0).astype(float)

    # 4) Range’ler
    for c in ["bus_stop_count_range", "train_stop_count_range", "poi_total_count_range", "poi_risk_score_range"]:
        to_int(df, c, default=0)
    for c in ["distance_to_bus_range", "distance_to_train_range", "distance_to_police_range", "distance_to_government_building_range"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            max_cat = int(s.max(skipna=True)) if pd.notna(s.max(skipna=True)) else 3
            df[c] = s.fillna(max_cat).round().astype("Int64")

    # 5) Nüfus (median)
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        median_pop = df["population"].median(skipna=True)
        df["population"] = df["population"].fillna(0 if pd.isna(median_pop) else median_pop)

    # 6) POI dominant type
    if "poi_dominant_type" in df.columns:
        df["poi_dominant_type"] = df["poi_dominant_type"].fillna("None").astype(str)

    # 7) Tarih normalize
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date

    # 7.5) Near-repeat (GEOID × kategori)
    try:
        if {"date","GEOID"}.issubset(df.columns):
            cat_col = "category_grouped" if "category_grouped" in df.columns else (
                      "subcategory_grouped" if "subcategory_grouped" in df.columns else
                      ("category" if "category" in df.columns else None))
            if cat_col:
                tmp = df[["date","GEOID",cat_col,"crime_count"]].copy()
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
                g = (tmp.groupby(["GEOID",cat_col,"date"], as_index=False)["crime_count"].sum())
                g["date"] = pd.to_datetime(g["date"])
                g = g.sort_values(["GEOID",cat_col,"date"])
                def _roll_counts(x):
                    x = x.set_index("date").asfreq("D", fill_value=0)
                    x["nr_7d"]  = x["crime_count"].rolling("7D").sum().shift(1)
                    x["nr_14d"] = x["crime_count"].rolling("14D").sum().shift(1)
                    return x.reset_index()
                g2 = (g.groupby(["GEOID", cat_col]).apply(_roll_counts)
                        .reset_index(level=[0,1]).reset_index(drop=True))
                g2["date"] = g2["date"].dt.date
                df = df.merge(g2[["GEOID",cat_col,"date","nr_7d","nr_14d"]],
                              on=["GEOID",cat_col,"date"], how="left")
                for c in ["nr_7d","nr_14d"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    except Exception as _e:
        print(f"near-repeat uyarı: {_e}")

    # 7.6) Komşu 7g toplam + 1g lag (opsiyonel)
    try:
        neighbors_path = Path(os.environ.get("NEIGHBOR_FILE", str(Path(DATA_DIR) / "neighbors.csv")))
        if neighbors_path.exists() and {"date","GEOID"}.issubset(df.columns):
            nbr = pd.read_csv(neighbors_path, dtype={"GEOID":str,"NEIGHBOR_GEOID":str})
            L = int(os.environ.get("GEOID_LEN","11"))
            for colx in ["GEOID","NEIGHBOR_GEOID"]:
                nbr[colx] = nbr[colx].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(L)
            daily = df[["date","GEOID","crime_count"]].copy()
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date
            daily = daily.groupby(["GEOID","date"], as_index=False)["crime_count"].sum()
            daily["date"] = pd.to_datetime(daily["date"])
            d2 = nbr.merge(daily.rename(columns={"GEOID":"NEIGHBOR_GEOID"}), on="NEIGHBOR_GEOID", how="left")
            d2 = d2.sort_values(["GEOID","date"])
            def _agg_nei(x):
                x = x.set_index("date").asfreq("D", fill_value=0)
                x["nei_7d_sum"] = x["crime_count"].rolling("7D").sum().shift(1)
                return x.reset_index()
            d3 = (d2.groupby("GEOID").apply(_agg_nei)
                    .reset_index(level=0).reset_index(drop=True))
            d3["date"] = d3["date"].dt.date
            d3 = d3.groupby(["GEOID","date"], as_index=False)["nei_7d_sum"].sum()
            df = df.merge(d3, on=["GEOID","date"], how="left")
            df["nei_7d_sum"] = pd.to_numeric(df["nei_7d_sum"], errors="coerce").fillna(0).astype(float)
    except Exception as _e:
        print(f"komşuluk özellik uyarı: {_e}")

    # 7.7) Tür-eşlemeli dışsal skor (opsiyonel)
    try:
        ext_map_path = Path(DATA_DIR) / "crime_type_externals_map.json"
        if ext_map_path.exists():
            with open(ext_map_path, "r", encoding="utf-8") as f:
                type_map = json.load(f)
            key_col = "category_grouped" if "category_grouped" in df.columns else (
                      "category" if "category" in df.columns else None)
            if key_col:
                def _ext_score(row):
                    cols = type_map.get(str(row[key_col]), [])
                    vals = []
                    for c in cols:
                        if c in df.columns:
                            try: vals.append(float(row.get(c, 0)))
                            except: pass
                    return float(np.nanmean(vals)) if len(vals)>0 else np.nan
                df["externals_type_score"] = df.apply(_ext_score, axis=1).fillna(0.0).astype(float)
    except Exception as _e:
        print(f"dışsal değişken uyarı: {_e}")

    preview_cols = [c for c in ["nr_7d","nr_14d","nei_7d_sum","externals_type_score"] if c in df.columns]
    if preview_cols:
        st.caption("🧩 Yeni mekânsal-zamansal özellikler (ilk 20 satır):")
        st.dataframe(df[preview_cols].head(20))

    outp = write_df(df, output_path)
    print(f"✅ {outp.name} kaydedildi. Satır sayısı: {len(df)}")
    return df

# =======================
# UI: 08 → 09 → Model
# =======================
st.title("📦 Günlük Suç Tahmin — Parquet Pipeline")

with st.sidebar:
    st.markdown("### GitHub Actions")
    persist = st.selectbox("Çıktıyı saklama modu", ["artifact", "commit", "none"], index=0,
                           help="artifact: repo’yu bozmadan sakla • commit: repo’ya yaz • none: sadece log")
    force_bypass = st.checkbox("07:00 kapısını yok say (force)", value=True)
    status_box = st.empty()
    _render_last_run_status(status_box)

    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("🚀 Actions’ta full pipeline"):
            if not (st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")):
                st.error("GH_TOKEN tanımlı değil.")
            else:
                try:
                    r = dispatch_workflow(
                        wf_selector=wf_selector,
                        persist=persist,
                        force=force_bypass,
                        top_k="50",
                        ref=ref_branch
                    )
                    if r["ok"]:
                        st.success("Tetiklendi!")
                    else:
                        st.error(f"Tetikleme başarısız: {r['status']} {r['text']}")
                    st.caption(f"workflow id={r.get('wf_id')} state={r.get('state')} path={r.get('path')} dispatch_in_file={r.get('has_dispatch')} ref={r.get('ref')}")
                except Exception as e:
                    st.error(f"Hata: {e}")

    with col_refresh:
        if st.button("📡 Son durumu yenile"):
            _render_last_run_status(status_box)

    with st.sidebar.expander("ACS Ayarları (Demografi)"):
        acs_year_default = os.environ.get("ACS_YEAR", "LATEST")
        whitelist_default = os.environ.get("DEMOG_WHITELIST", "")
        level_default = os.environ.get("CENSUS_GEO_LEVEL", "auto")
        acs_year_in = st.text_input("ACS_YEAR (LATEST veya YYYY)", value=str(acs_year_default or "LATEST"))
        whitelist_in = st.text_input("DEMOG_WHITELIST (virgüllü; boş = hepsi)",
                                     value=str(whitelist_default or ""))
        levels = ["auto", "tract", "blockgroup", "block"]
        try:
            idx = levels.index(level_default) if level_default in levels else 0
        except Exception:
            idx = 0
        level_in = st.selectbox("CENSUS_GEO_LEVEL", levels, index=idx)
        os.environ["CENSUS_GEO_LEVEL"] = level_in

        pop_default = os.environ.get("POPULATION_PATH", str(POPULATION_PATH))
        pop_url_in = st.text_input("POPULATION_PATH (yerel CSV yolu)", value=str(pop_default or ""))
        _v = str(acs_year_in).strip()
        os.environ["ACS_YEAR"] = "LATEST" if _v.upper()=="LATEST" else (re.sub(r"\D","",_v) if len(re.sub(r"\D","",_v))==4 else "LATEST")
        os.environ["DEMOG_WHITELIST"] = str(whitelist_in or "")
        if re.match(r"^https?://", str(pop_url_in), flags=re.I):
            st.error("CSV-only mod: URL kabul edilmez. Yerel bir CSV yolu girin.")
        else:
            os.environ["POPULATION_PATH"] = pop_url_in or str(POPULATION_PATH)

    with st.sidebar.expander("Workflow ayarları", True):
        # Dosya adı (path) veya workflow 'name' ya da ID kabul ediyor
        wf_default = os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")
        wf_selector = st.text_input("Workflow (ad / path / id)", value=wf_default, help="Örn: full_pipeline.yml veya 'Full SF Crime Pipeline'")
    
        # Hangi branche dispatch edilecek?
        ref_default = os.environ.get("GITHUB_REF_NAME", "main")
        ref_branch = st.text_input("Ref/branch", value=ref_default)


# 3) sf_crime_08 (Parquet göster)
st.markdown("### 3) Güncel sf_crime_08 (ilk 20 satır)")
df08 = load_sf_crime_08(DATA_DIR / "sf_crime_08.csv")
if df08 is not None:
    st.dataframe(df08.head(20))
    # 09’u Parquet yaz
    clean_and_save_crime_09(df08, DATA_DIR / "sf_crime_09.parquet")
    st.success("✅ sf_crime_09.parquet kaydedildi.")
else:
    st.info("Henüz sf_crime_08 bulunamadı. Pipeline’ı çalıştırabilir veya artifact erişimini ayarlayabilirsiniz.")

# 4) sf_crime_09 göster
try:
    df09 = read_df(DATA_DIR / "sf_crime_09.parquet")
    st.markdown("### 4) Güncel sf_crime_09 (ilk 20 satır)")
    st.dataframe(df09.head(20))
except Exception as e:
    st.warning(f"sf_crime_09 okunamadı: {e}")
    df09 = None

# 5) Hızlı Model (Parquet I/O)
if df09 is not None:
    st.markdown("### 5) Hızlı Model (ZI/Hurdle + Quantile + Kalibrasyon)")
    if st.button("🧠 Modeli Eğit (örnek)"):
        deps = _load_ml_deps()
        np = deps["np"]; shap = deps["shap"]
        TimeSeriesSplit = deps["TimeSeriesSplit"]
        roc_auc_score = deps["roc_auc_score"]; brier_score_loss = deps["brier_score_loss"]; mean_absolute_error = deps["mean_absolute_error"]
        CalibratedClassifierCV = deps["CalibratedClassifierCV"]
        PartialDependenceDisplay = deps["PartialDependenceDisplay"]
        LGBMClassifier = deps["LGBMClassifier"]; LGBMRegressor = deps["LGBMRegressor"]
        LimeTabularExplainer = deps["LimeTabularExplainer"]

        if "date" in df09.columns:
            df09 = df09.sort_values("date").reset_index(drop=True)

        y_occ = (df09["crime_count"] > 0).astype(int)
        y_cnt = df09.loc[y_occ == 1, "crime_count"]

        feat_cols = [c for c in df09.columns if c not in ["crime_count","category","subcategory",
                                                          "category_grouped","subcategory_grouped","date","datetime"]]
        X_all = df09[feat_cols].select_dtypes(include=[np.number]).fillna(0.0)
        X_occ = X_all
        X_cnt = X_all.loc[y_occ == 1]

        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, test_idx = list(tscv.split(X_occ))[-1]

        base_clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1,
                                  class_weight="balanced", random_state=42)
        clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=3)
        clf.fit(X_occ.iloc[train_idx], y_occ.iloc[train_idx])
        p_hat = clf.predict_proba(X_occ.iloc[test_idx])[:, 1]
        st.write("Varlık modeli AUC:", float(roc_auc_score(y_occ.iloc[test_idx], p_hat)))
        st.write("Brier:", float(brier_score_loss(y_occ.iloc[test_idx], p_hat)))

        quantiles = [0.1, 0.5, 0.9]
        q_models = {}
        for q in quantiles:
            qr = LGBMRegressor(objective="quantile", alpha=q,
                               n_estimators=400, learning_rate=0.05, random_state=42)
            tr_idx_pos = X_cnt.index.intersection(X_occ.index[train_idx])
            te_idx_pos = X_cnt.index.intersection(X_occ.index[test_idx])
            qr.fit(X_cnt.loc[tr_idx_pos], y_cnt.loc[tr_idx_pos])
            if q == 0.5:
                pred_med = qr.predict(X_cnt.loc[te_idx_pos])
                st.write("Pozitiflerde MAE (median):",
                         float(mean_absolute_error(y_cnt.loc[te_idx_pos], pred_med)))
            q_models[q] = qr

        if 0.5 in q_models:
            mu_med = q_models[0.5].predict(X_all)
            p_all = clf.predict_proba(X_all)[:, 1]
            df09["pred_p_occ"] = p_all
            df09["pred_q10"] = q_models[0.1].predict(X_all) if 0.1 in q_models else np.nan
            df09["pred_q50"] = mu_med
            df09["pred_q90"] = q_models[0.9].predict(X_all) if 0.9 in q_models else np.nan
            df09["pred_expected"] = p_all * np.maximum(mu_med, 0)
            st.dataframe(df09[["pred_p_occ","pred_q10","pred_q50","pred_q90","pred_expected"]].head(20))
            out_pred = write_df(df09, DATA_DIR / "sf_crime_09_with_preds.parquet")
            st.success(f"✔ Tahminli dosya kaydedildi: {out_pred.name}")

        # Açıklanabilirlik
        st.markdown("### 6) Açıklanabilirlik (global & local)")
        clf_tree = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1,
                                  class_weight="balanced", random_state=42)
        clf_tree.fit(X_occ.iloc[train_idx], y_occ.iloc[train_idx])

        expl_clf = shap.TreeExplainer(clf_tree)
        sample_idx = X_occ.iloc[test_idx].sample(min(1500, len(test_idx)), random_state=42).index
        shap_vals_cls = expl_clf.shap_values(X_occ.loc[sample_idx])
        shap_pos = shap_vals_cls[1] if isinstance(shap_vals_cls, list) else shap_vals_cls

        cat_col = ("category_grouped" if "category_grouped" in df09.columns else
                   ("subcategory_grouped" if "subcategory_grouped" in df09.columns else None))
        tabs = st.tabs([
            "Global (Sınıf: var/yok)",
            "Global (Sayı: kuantil)",
            "Local (satır)",
            "PDP / ICE",
            "Rapor (mini kart)",
        ])

        # (opsiyonel) forensic rapor
        try:
            with st.spinner("📑 Rapor hazırlanıyor..."):
                res = build_forensic_report(
                    df09=df09, X_occ=X_occ, X_cnt=X_cnt, y_occ=y_occ, y_cnt=y_cnt,
                    train_idx=train_idx, test_idx=test_idx,
                    p_hat=p_hat, q_models=q_models,
                    pred_med=(pred_med if 'pred_med' in locals() else None),
                    base_clf=base_clf, clf_tree=clf_tree,
                    DATA_DIR=DATA_DIR,
                    out_pred=(str(DATA_DIR / "sf_crime_09_with_preds.parquet"))
                )
        except Exception as e:
            st.error(f"Rapor oluşturulurken hata: {e}")
            res = None

        if res:
            st.success(f"Rapor hazır: {res.get('dir')}")
            colA, colB, colC = st.columns(3)
            with colA:
                if res.get("pdf") and Path(res["pdf"]).exists():
                    st.download_button("📄 PDF indir", Path(res["pdf"]).read_bytes(), file_name="run_report.pdf")
                st.download_button("📊 Metrikler (CSV)", Path(res["metrics_csv"]).read_bytes(), file_name="metrics.csv")
            with colB:
                st.download_button("🔎 SHAP (top20 CSV)", Path(res["shap_csv"]).read_bytes(), file_name="global_shap_top20.csv")
                st.download_button("🔥 Top-100 riskli hücre", Path(res["top100_csv"]).read_bytes(), file_name="top100_risky_cells.csv")
            with colC:
                st.download_button("🧾 Forensic JSON", Path(res["forensic_json"]).read_bytes(), file_name="forensic_log.json")
        else:
            st.info("Forensic rapor üretimi atlandı veya başarısız.")

        # -------------------------------
        # Global — Sınıf (var/yok)
        # -------------------------------
        with tabs[0]:
            st.caption("Pozitif sınıf (Y>0) için ortalama mutlak SHAP — ilk 10")
            mean_abs = np.abs(shap_pos).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]
            top_feat = X_occ.columns[top_idx]
            top_vals = mean_abs[top_idx]
            top_df = pd.DataFrame({"özellik": top_feat, "önem(Mean|SHAP|)": top_vals})
            st.dataframe(top_df, use_container_width=True)
            st.bar_chart(top_df.set_index("özellik"))

            cat_col2 = "category_grouped" if "category_grouped" in df09.columns else (
                       "subcategory_grouped" if "subcategory_grouped" in df09.columns else None)
            if cat_col2:
                col1, col2 = st.columns(2)
                with col1:
                    pick1 = st.selectbox("Sınıf 1 (ör. Theft)", sorted(df09[cat_col2].dropna().unique()))
                with col2:
                    pick2 = st.selectbox("Sınıf 2 (ör. Assault)", sorted(df09[cat_col2].dropna().unique()))
                for pick in [pick1, pick2]:
                    sub_idx = df09.loc[sample_idx][df09.loc[sample_idx, cat_col2] == pick].index
                    if len(sub_idx) >= 20:
                        shap_sub = expl_clf.shap_values(X_occ.loc[sub_idx])
                        shap_sub_pos = shap_sub[1] if isinstance(shap_sub, list) else shap_sub
                        mabs = np.abs(shap_sub_pos).mean(axis=0)
                        top_idx2 = np.argsort(mabs)[::-1][:10]
                        top_df2 = pd.DataFrame({
                            "özellik": X_occ.columns[top_idx2],
                            f"{pick} için önem": mabs[top_idx2]
                        })
                        st.markdown(f"**{pick} — ilk 10 etken**")
                        st.dataframe(top_df2, use_container_width=True)
                    else:
                        st.info(f"{pick} için yeterli örnek yok (≥20 önerilir).")

        # -------------------------------
        # Global — Sayı (kuantil)
        # -------------------------------
        with tabs[1]:
            if 0.5 in q_models:
                st.caption("Kuantil (q=0.5) LightGBMRegressor için SHAP — ilk 10")
                expl_reg = shap.TreeExplainer(q_models[0.5])
                sample_idx_reg = X_cnt.sample(min(1500, len(X_cnt)), random_state=42).index
                shap_vals_reg = expl_reg.shap_values(X_cnt.loc[sample_idx_reg])
                mean_abs_r = np.abs(shap_vals_reg).mean(axis=0)
                top_idx_r = np.argsort(mean_abs_r)[::-1][:10]
                top_df_r = pd.DataFrame({
                    "özellik": X_cnt.columns[top_idx_r],
                    "önem(Mean|SHAP|)": mean_abs_r[top_idx_r]
                })
                st.dataframe(top_df_r, use_container_width=True)
                st.bar_chart(top_df_r.set_index("özellik"))
            else:
                st.info("Kuantil regresyon modeli bulunamadı.")

        # -------------------------------
        # Local — tek satır
        # -------------------------------
        with tabs[2]:
            st.caption("Seçilen satır için olasılık ve katkılar")
            idx = st.number_input("Satır indexi", min_value=0, max_value=int(len(X_all)-1), value=0, step=1)
            x_row = X_all.iloc[[idx]]
            p_row = float(clf.predict_proba(x_row)[:,1])
            exp_row = float(df09.loc[x_row.index, "pred_expected"]) if "pred_expected" in df09.columns else np.nan
            st.write(f"**P(Y>0)** = {p_row:.3f}  |  **Beklenen sayı** ≈ {exp_row:.2f}")
            shap_row = expl_clf.shap_values(x_row)
            shap_row_pos = shap_row[1][0] if isinstance(shap_row, list) else shap_row[0]
            contrib = pd.DataFrame({
                "özellik": x_row.columns,
                "değer": x_row.iloc[0].values,
                "katkı(SHAP)": shap_row_pos
            }).sort_values("katkı(SHAP)", key=np.abs, ascending=False).head(15)
            st.dataframe(contrib, use_container_width=True)

        # -------------------------------
        # PDP / ICE
        # -------------------------------
        with tabs[3]:
            st.caption("PDP (Sınıf modeli, target=1)")
            candidates = [c for c in ["event_hour","nei_7d_sum","nr_7d","bus_stop_count","poi_risk_score"] if c in X_occ.columns]
            feats = st.multiselect("PDP özellik(ler)i", options=candidates, default=candidates[:2])
            if len(feats) > 0:
                for f in feats[:3]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    PartialDependenceDisplay.from_estimator(
                        clf_tree, X_occ, [f], kind="average", target=1, ax=ax
                    )
                    ax.set_title(f"PDP — {f}")
                    st.pyplot(fig, clear_figure=True)
            else:
                st.info("Listeden en az bir özellik seç.")

        with tabs[4]:
            st.caption("İki sınıf için özet kart (global SHAP top-5)")
            if cat_col:
                pick_a = st.selectbox("Kart A sınıfı", sorted(df09[cat_col].dropna().unique()), key="cardA")
                pick_b = st.selectbox("Kart B sınıfı", sorted(df09[cat_col].dropna().unique()), key="cardB")
                def _topk_for_class(pick, k=5):
                    sub_idx = df09.loc[sample_idx][df09.loc[sample_idx, cat_col] == pick].index
                    if len(sub_idx) < 20:
                        return pd.DataFrame({"özellik": [], "önem": []})
                    shap_sub = expl_clf.shap_values(X_occ.loc[sub_idx])
                    shap_sub_pos = shap_sub[1] if isinstance(shap_sub, list) else shap_sub
                    mabs = np.abs(shap_sub_pos).mean(axis=0)
                    top = np.argsort(mabs)[::-1][:k]
                    return pd.DataFrame({"özellik": X_occ.columns[top], "önem": mabs[top]})
                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"**{pick_a} — Top 5 etken**")
                    st.dataframe(_topk_for_class(pick_a), use_container_width=True)
                with colB:
                    st.markdown(f"**{pick_b} — Top 5 etken**")
                    st.dataframe(_topk_for_class(pick_b), use_container_width=True)
            else:
                st.info("category_grouped / subcategory_grouped yoksa sınıf kartları oluşturulamaz.")
