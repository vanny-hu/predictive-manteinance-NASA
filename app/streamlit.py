import sys
import os

# ✅ PATH SETUP
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(BASE_DIR, "src")

sys.path.append(SRC_PATH)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from function import load_data, add_features

# ✅ PATH DATI E MODELLO
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ✅ HEADER
st.title("🚀 Predictive Maintenance Dashboard")

# ✅ SELEZIONE FILE VALIDI
files = [
    f for f in os.listdir(DATA_PATH)
    if f.startswith("test") and f.endswith(".txt")
]

file = st.selectbox("📂 Select dataset", files)

# ✅ CONTROLLO MODELLO
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run training first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ✅ LOAD DATA (path relativo)
df = load_data(os.path.join("data/raw", file))

# ✅ FEATURE ENGINEERING
df = add_features(df)

# ✅ SELEZIONE UNIT
if "unit" in df.columns:
    unit = st.selectbox("🏭 Select Unit (Machine)", sorted(df["unit"].unique()))
    df = df[df["unit"] == unit]
    df = df.tail(50)

# ✅ SENSOR TREND (mantiene st.line_chart)
st.subheader("📊 Sensor Trend")

sensor_cols = [col for col in df.columns if "sensor" in col]
sensor = st.selectbox("🔍 Select sensor", sensor_cols)

# ✅ etichette assi (workaround)
st.markdown(f"**Y-axis:** {sensor} (Sensor Value)")
st.line_chart(df.set_index("cycle")[sensor])
st.markdown("**X-axis:** Cycle (Time)")

# ✅ PREPARAZIONE INPUT MODELLO
X = df.drop(["unit", "cycle", "RUL", "label"], axis=1, errors="ignore")

# ✅ PREDIZIONE
pred = model.predict(X)

# ✅ OUTPUT
st.subheader("🔮 Predictions")
st.write(pred[:20])

# ✅ METRICA
fail_rate = pred.mean()
st.metric("⚠️ Failure Probability", f"{fail_rate:.2f}")

# ✅ ALERT
if fail_rate > 0.5:
    st.error("🚨 High failure risk detected!")
else:
    st.success("✅ System operating normally")

# ✅ DISTRIBUZIONE PREDIZIONI (con assi)
st.subheader("📈 Prediction Distribution")

fig, ax = plt.subplots(figsize=(6, 4))

counts = pd.Series(pred).value_counts().sort_index()

counts.plot(
    kind="bar",
    ax=ax,
    color=["green" if i == 0 else "red" for i in counts.index]
)

ax.set_title("Prediction Distribution")
ax.set_xlabel("Class (0 = Healthy, 1 = Failure Risk)")
ax.set_ylabel("Count")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Healthy", "Failure"])

ax.grid(axis="y")

st.pyplot(fig)