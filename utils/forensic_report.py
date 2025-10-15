# utils/forensic_report.py
from pathlib import Path
from datetime import datetime
import json, hashlib, platform, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def build_forensic_report(
    *, df09: pd.DataFrame,
    X_occ: pd.DataFrame, X_cnt: pd.DataFrame,
    y_occ: pd.Series, y_cnt: pd.Series,
    train_idx, test_idx,
    p_hat: np.ndarray,
    q_models: dict,
    pred_med: np.ndarray | None,
    base_clf,           # LGBMClassifier (param/log için)
    clf_tree,           # SHAP için yeniden eğitilen ağaç sınıflandırıcı
    DATA_DIR: Path,
    out_pred: Path
) -> dict:
    run_id  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rep_dir = DATA_DIR / "reports" / run_id
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1) Kalibrasyon eğrisi
    prob_true, prob_pred = calibration_curve(y_occ.iloc[test_idx], p_hat, n_bins=15, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0,1],[0,1], "--", label="Mükemmel")
    ax.set_xlabel("Tahmin olasılığı"); ax.set_ylabel("Gözlenen frekans"); ax.legend(loc="lower right")
    fig.tight_layout()
    calib_png = rep_dir / "calibration_curve.png"
    fig.savefig(calib_png, dpi=150); plt.close(fig)

    # 2) Global SHAP (sınıf modeli, test fold örneklem)
    expl_clf = shap.TreeExplainer(clf_tree)
    sample_idx = X_occ.iloc[test_idx].sample(min(1500, len(test_idx)), random_state=42).index
    shap_vals_cls = expl_clf.shap_values(X_occ.loc[sample_idx])
    shap_pos = shap_vals_cls[1] if isinstance(shap_vals_cls, list) else shap_vals_cls

    mean_abs = np.abs(shap_pos).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:20]
    shap_top_cls = pd.DataFrame({"feature": X_occ.columns[top_idx], "importance": mean_abs[top_idx]})
    shap_top_cls.to_csv(rep_dir / "global_shap_classifier_top20.csv", index=False)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.barh(shap_top_cls["feature"][::-1], shap_top_cls["importance"][::-1])
    ax2.set_title("Global SHAP (classifier, top 20)")
    fig2.tight_layout()
    fig2.savefig(rep_dir / "global_shap_classifier_top20.png", dpi=150); plt.close(fig2)

    # 3) Global SHAP (q=0.5) varsa
    if 0.5 in q_models:
        expl_reg = shap.TreeExplainer(q_models[0.5])
        sample_idx_reg = X_cnt.sample(min(1500, len(X_cnt)), random_state=42).index
        shap_vals_reg = expl_reg.shap_values(X_cnt.loc[sample_idx_reg])
        mean_abs_r = np.abs(shap_vals_reg).mean(axis=0)
        top_idx_r  = np.argsort(mean_abs_r)[::-1][:20]
        shap_top_reg = pd.DataFrame({"feature": X_cnt.columns[top_idx_r], "importance": mean_abs_r[top_idx_r]})
        shap_top_reg.to_csv(rep_dir / "global_shap_reg_q50_top20.csv", index=False)

    # 4) Top-100 riskli hücre
    topk = df09[["GEOID","date","pred_expected"]].sort_values("pred_expected", ascending=False).head(100)
    topk.to_csv(rep_dir / "top100_risky_cells.csv", index=False)

    # 5) Metrikler (AUC/Brier + MAE/SMAPE)
    from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
    def _smape(a, f, eps=1e-9):
        a = np.asarray(a, dtype=float); f = np.asarray(f, dtype=float)
        return float(np.mean(2.0*np.abs(f-a)/(np.abs(a)+np.abs(f)+eps)))
    auc   = float(roc_auc_score(y_occ.iloc[test_idx], p_hat))
    brier = float(brier_score_loss(y_occ.iloc[test_idx], p_hat))
    mae_q50   = float(mean_absolute_error(y_cnt.loc[X_cnt.index.intersection(X_occ.index[test_idx])], pred_med)) if pred_med is not None else None
    smape_q50 = _smape(y_cnt.loc[X_cnt.index.intersection(X_occ.index[test_idx])], pred_med) if pred_med is not None else None

    metrics_df = pd.DataFrame({
        "metric": ["AUC","Brier","MAE_q50","SMAPE_q50"],
        "value":  [auc, brier, mae_q50, smape_q50]
    })
    metrics_df.to_csv(rep_dir / "metrics.csv", index=False)

    # 6) Forensic JSON
    forensic = {
        "run_id": run_id,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "data": {
            "input_csv": str(DATA_DIR / "sf_crime_09.csv"),
            "preds_csv": str(out_pred),
            "hash_input": _sha256(DATA_DIR / "sf_crime_09.csv") if (DATA_DIR / "sf_crime_09.csv").exists() else None,
            "hash_preds": _sha256(out_pred),
        },
        "model": {
            "classifier_params": base_clf.get_params(),
            "calibration": "isotonic",
            "regressor_quantiles": sorted([float(k) for k in q_models.keys()]),
            "split": {"type": "TimeSeriesSplit", "n_splits": 3, "test_fold_len": int(len(test_idx))}
        },
        "features": list(X_occ.columns),
        "metrics": {"auc": auc, "brier": brier, "mae_q50": mae_q50, "smape_q50": smape_q50},
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "shap": shap.__version__,
        }
    }
    (rep_dir / "forensic_log.json").write_text(json.dumps(forensic, indent=2), encoding="utf-8")

    # 7) PDF (varsa reportlab)
    pdf_path = None
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        pdf_path = rep_dir / "run_report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4; y = h - 2*cm
        c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, y, f"Crime Model Report — {run_id}"); y -= 1.0*cm
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, y, f"AUC={auc:.3f} | Brier={brier:.3f} | MAE(q50)={mae_q50 if mae_q50 is not None else 'NA'} | SMAPE(q50)={smape_q50 if smape_q50 is not None else 'NA'}"); y -= 0.7*cm
        if calib_png.exists():
            c.drawImage(str(calib_png), 2*cm, y-8*cm, width=12*cm, height=8*cm); y -= 9*cm
        c.drawString(2*cm, y, "Ekler: global_shap_*.csv/png, top100_risky_cells.csv, metrics.csv, forensic_log.json")
        c.showPage(); c.save()
    except Exception:
        pass

    # 'latest' kopyası
    latest_dir = DATA_DIR / "reports" / "latest"
    try:
        import shutil
        if latest_dir.exists(): shutil.rmtree(latest_dir)
        shutil.copytree(rep_dir, latest_dir)
    except Exception:
        pass

    return {
        "run_id": run_id,
        "dir": rep_dir,
        "pdf": pdf_path,
        "calibration_png": calib_png,
        "metrics_csv": rep_dir / "metrics.csv",
        "shap_csv": rep_dir / "global_shap_classifier_top20.csv",
        "top100_csv": rep_dir / "top100_risky_cells.csv",
        "forensic_json": rep_dir / "forensic_log.json",
        "latest_dir": latest_dir
    }
