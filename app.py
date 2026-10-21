"""Local model/results viewer: streamlit run app.py."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TCM Syndrome Classification", layout="wide")
st.title("TCM syndrome classification")
st.caption("DS 340W • Jackson Shi • Research model")
st.write("**Research question:** How accurately can machine learning classify TCM syndromes from symptoms, and does feature selection improve a multi-output Random Forest?")
st.code("Symptom records → source/duplicate grouping → 70/20/10 split\nTraining → TF-IDF → optional PCC-MLRF-style selection → classifier\nTraining CV → choose threshold and model → development test\nFinal 10% → reserved until approach is frozen", language="text")
root = st.text_input("Results directory", "runs")
folders = sorted(p.parent for p in Path(root).glob("*/run.json"))
if not folders:
    st.info("Train first: python -m tcm train --dataset stomach")
    st.stop()
folder = st.selectbox("Experiment", folders, format_func=str)
meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
st.write("Selected model:", meta["winner"], " · Final validation:", meta["validation_status"])
st.dataframe(pd.read_csv(folder / "comparison.csv"), hide_index=True)
st.image(str(folder / "comparison.png"))
if (folder / "final_validation.json").exists():
    with st.expander("Final validation results"):
        st.json(json.loads((folder / "final_validation.json").read_text(encoding="utf-8")))
with st.expander("Data source and experiment details"):
    st.json(meta)
st.subheader("Try symptom text")
text = st.text_area("Use Chinese symptom phrases, separated by 、 or commas", "腹部冷痛、腹泻、食欲不振")
if st.button("Predict syndrome labels"):
    if not text.strip():
        st.warning("Enter symptom text first.")
    else:
        bundle = joblib.load(folder / "model.joblib")
        model = bundle["model"]
        row = model.scores([text])[0]
        frame = pd.DataFrame({"syndrome": model.classes, "model_score": row, "above_threshold": row >= bundle["threshold"]})
        st.dataframe(frame.sort_values("model_score", ascending=False).head(15), hide_index=True)
        st.caption(f"Decision threshold: {bundle['threshold']}. Scores are not calibrated clinical probabilities. Input should match the training vocabulary and language.")
