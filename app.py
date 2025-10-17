# streamlit_app.py
import requests, streamlit as st

OWNER = "cem5113"
REPO = "crime_prediction_data_pre"
WORKFLOW = "full_pipeline.yml"  # .github/workflows içindeki dosya adı
REF = "main"

st.title("SF Crime – Minimal Trigger")

if st.button("Pipeline’ı Başlat"):
    token = st.secrets["GH_TOKEN"]
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WORKFLOW}/dispatches"
    payload = {"ref": REF, "inputs": {"persist": "artifact", "force": "true", "top_k": "50"}}
    r = requests.post(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }, json=payload, timeout=30)
    st.write("OK" if r.status_code in (201, 204) else f"Error: {r.status_code} {r.text}")
