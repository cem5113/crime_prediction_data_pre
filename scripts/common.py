# scripts/common.py
from pathlib import Path
import pandas as pd
import numpy as np
import os, re

def clean_and_save_crime_09(
    input_obj: str | pd.DataFrame,
    output_path: str
):
    """
    08 → 09 normalize + impute
    - hour_range normalize, GEOID pad/kes, dtype sıkılaştırma, kolon sırası.
    - Hem DataFrame hem de yol (chunk) girişi desteklenir.
    """

    def _norm_geoid(s: pd.Series, target_len: int) -> pd.Series:
        s = s.astype(str).str.extract(r"(\d+)")[0]
        return s.fillna("").str.zfill(target_len).str[-target_len:]

    def _normalize_hour_range(s: pd.Series, fallback_from_event_hour: pd.Series | None = None) -> pd.Series:
        def to_hr(x):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(x))
            if m:
                a, b = int(m.group(1)) % 24, int(m.group(2)) % 24
                return f"{a:02d}-{b:02d}"
            return np.nan
        out = s.astype(str).apply(to_hr)
        if out.isna().any() and fallback_from_event_hour is not None:
            eh = pd.to_numeric(fallback_from_event_hour, errors="coerce").fillna(0).astype(int) % 24
            start = (eh // 3) * 3
            end = (start + 3) % 24
            hr = start.astype(str).str.zfill(2) + "-" + end.astype(str).str.zfill(2)
            out = out.fillna(hr)
        return out.fillna("00-03")

    def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
        # date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "datetime" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date

        # event_hour & hour_range
        if "event_hour" in df.columns:
            df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int) % 24
        if "hour_range" in df.columns:
            df["hour_range"] = _normalize_hour_range(df["hour_range"], df.get("event_hour"))
        else:
            df["hour_range"] = _normalize_hour_range(pd.Series(np.nan, index=df.index), df.get("event_hour"))

        # GEOID
        geoid_len = int(os.getenv("GEOID_LEN", "11"))
        if "GEOID" in df.columns:
            df["GEOID"] = _norm_geoid(df["GEOID"], geoid_len)

        # sayaçlar → int32
        int_cols = [
            "crime_count",
            "911_request_count_hour_range",
            '911_request_count_daily(before_24_hours)',
            "311_request_count",
            "bus_stop_count",
            "train_stop_count",
            "poi_total_count",
        ]
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")

        # risk skoru → float32
        if "poi_risk_score" in df.columns:
            df["poi_risk_score"] = pd.to_numeric(df["poi_risk_score"], errors="coerce").fillna(0.0).astype("float32")

        # binary → int8
        for c in ["is_near_police", "is_near_government"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].replace({True: 1, False: 0}), errors="coerce").fillna(0).astype("int8")

        # mesafeler → float32
        for c in ["distance_to_bus", "distance_to_train", "distance_to_police", "distance_to_government_building"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(9999.0).astype("float32")

        # range kolonları → int16 + category
        range_cols = [
            "bus_stop_count_range",
            "train_stop_count_range",
            "poi_total_count_range",
            "poi_risk_score_range",
            "distance_to_bus_range",
            "distance_to_train_range",
            "distance_to_police_range",
            "distance_to_government_building_range",
        ]
        for c in range_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int16").astype("category")

        # nüfus → int32
        if "population" in df.columns:
            df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype("int32")

        # POI dominant type → category
        if "poi_dominant_type" in df.columns:
            df["poi_dominant_type"] = df["poi_dominant_type"].fillna("None").astype("category")

        # kolon sırası
        tail = [c for c in ["date", "hour_range", "GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in tail] + tail
        return df[cols]

    # DF girdisi
    if isinstance(input_obj, pd.DataFrame):
        df = _normalize_chunk(input_obj.copy())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ {output_path} kaydedildi. Satır: {len(df)}")
        return df

    # Yol girdisi (chunk)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk in pd.read_csv(input_obj, chunksize=200_000, dtype={"GEOID": str}, low_memory=False):
        chunk = _normalize_chunk(chunk)
        chunk.to_csv(output_path, index=False, mode=("w" if first else "a"), header=first)
        first = False
    print(f"✅ {output_path} kaydedildi (chunk).")
    return None
