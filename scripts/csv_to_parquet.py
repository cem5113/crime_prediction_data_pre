#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV/ZIP → Parquet dönüştürücü (Polars tabanlı)

- Girdi: tek .csv, .csv.gz, klasör (içindeki tüm .csv/.csv.gz), veya .zip (içindeki tüm .csv/.csv.gz)
- Çıktı: aynı adlarla .parquet dosyaları
- Sıkıştırma: zstd (önerilir), snappy, uncompressed
- İstatistik: --stats ile sütun istatistiklerini Parquet metadata’sına yazar

Kullanım:
  python scripts/csv_to_parquet.py \
    --input artifacts/ \
    --output parquet_out/ \
    --compression zstd \
    --stats
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
import pathlib
from typing import List, Optional

try:
    import polars as pl
except Exception as e:
    print("❌ 'polars' gerekli. Lütfen 'pip install polars' kurun.")
    raise

# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #

def _is_csv_like(p: pathlib.Path) -> bool:
    """*.csv veya *.csv.gz dosyası mı? (case-insensitive)"""
    name = p.name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz")

def _safe_extract_zip_csvs(zippath: pathlib.Path) -> List[pathlib.Path]:
    """
    ZIP içindeki *.csv / *.csv.gz dosyalarını güvenli şekilde temp klasöre çıkarır
    ve çıkarılan dosya yollarını döndürür.
    """
    outdir = pathlib.Path(tempfile.mkdtemp(prefix="csv2pq_"))
    with zipfile.ZipFile(zippath, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".csv", ".csv.gz"))]
        if not members:
            print(f"[WARN] ZIP içinde CSV yok: {zippath}")
            return []
        for m in members:
            # path traversal önleme: sadece basename ile yaz
            dest = outdir / pathlib.Path(m).name
            with zf.open(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    # Hem .csv hem .csv.gz topla
    return sorted(list(outdir.rglob("*.csv"))) + sorted(list(outdir.rglob("*.csv.gz")))

def list_csvs(input_path: str, exclude_dir: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    """
    input_path: dosya (.csv/.csv.gz/.zip) ya da klasör
    exclude_dir: (opsiyonel) bu klasörün altında kalan yolları es geç (örn. output klasörü)
    """
    p = pathlib.Path(input_path)
    if p.is_file():
        low = p.name.lower()
        if low.endswith(".csv") or low.endswith(".csv.gz"):
            return [p]
        if low.endswith(".zip"):
            return _safe_extract_zip_csvs(p)
        raise FileNotFoundError(f"Desteklenmeyen dosya türü: {p}")

    if p.is_dir():
        csvs: List[pathlib.Path] = []
        for q in p.rglob("*"):
            if not q.is_file():
                continue
            if not _is_csv_like(q):
                continue
            if exclude_dir and exclude_dir in q.parents:
                # çıktı klasörü altındakileri alma
                continue
            csvs.append(q)
        return sorted(csvs)

    raise FileNotFoundError(f"Girdi bulunamadı: {input_path}")

def csv_to_parquet(
    csv_path: pathlib.Path,
    out_dir: pathlib.Path,
    compression: str,
    write_stats: bool,
    row_group_size: int = 128_000,
) -> pathlib.Path:
    """
    Tek bir CSV'yi Parquet'e dönüştürür. Parquet yolu döndürür.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # .csv.gz gibi çift uzantıları düzgün kesmek için:
    stem = csv_path.name
    if stem.lower().endswith(".csv.gz"):
        stem = stem[:-7]  # .csv.gz kaldır
    elif stem.lower().endswith(".csv"):
        stem = stem[:-4]  # .csv kaldır

    out_path = out_dir / f"{stem}.parquet"
    print(f"[INFO] {csv_path} → {out_path}")

    # Lazy okuma → direkt Parquet’e akıt (RAM dostu)
    lf = pl.scan_csv(
        str(csv_path),
        has_header=True,
        ignore_errors=True,        # bozuk satırlar varsa yut
        try_parse_dates=True,
        infer_schema_length=10000, # daha sağlam tür kestirimi
        low_memory=True,
    )

    lf.sink_parquet(
        str(out_path),
        compression=compression,   # "zstd" (önerilir) ya da "snappy" / "uncompressed"
        statistics=write_stats,    # sütun istatistikleri (predicate pushdown için)
        row_group_size=row_group_size,
    )
    return out_path

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="CSV/ZIP → Parquet dönüştürücü (Polars)")
    ap.add_argument("--input", required=True, help="CSV/ZIP yolu veya klasör")
    ap.add_argument("--output", required=True, help="Parquet çıktı klasörü")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "uncompressed"],
                    help="Parquet sıkıştırma (varsayılan: zstd)")
    ap.add_argument("--stats", action="store_true", help="Parquet sütun istatistiklerini yaz")
    ap.add_argument("--row-group-size", type=int, default=128_000,
                    help="Parquet row group boyutu (satır). Varsayılan: 128k")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output).resolve()
    csv_files = list_csvs(args.input, exclude_dir=out_dir)

    if not csv_files:
        print(f"[WARN] Hiç CSV bulunamadı: {args.input}")
        sys.exit(0)

    n_ok, n_fail = 0, 0
    for csvf in csv_files:
        try:
            csv_to_parquet(
                csv_path=csvf,
                out_dir=out_dir,
                compression=args.compression,
                write_stats=args.stats,
                row_group_size=args.row_group_size,
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[ERROR] {csvf}: {e}")

    print(f"[OK] Tamamlandı. Başarılı: {n_ok}, Hatalı: {n_fail}. Çıktı: {out_dir}")

if __name__ == "__main__":
    main()
