import streamlit as st
import matplotlib.pyplot as plt
from model import train_and_predict

st.set_page_config(
    page_title="Crash Assistant PRO",
    layout="wide"
)

st.title("🚀 Crash Assistant PRO")

# ===== Sidebar: Add new value =====
st.sidebar.header("➕ Ajouter un résultat")

new_value = st.sidebar.number_input(
    "Crash value",
    min_value=1.0,
    step=0.01
)

if st.sidebar.button("Add"):
    with open("data.csv", "a") as f:
        f.write(f"\n{new_value}")
    st.sidebar.success("Ajouté avec succès ✅")

# ===== AI =====
prediction, confidence, df = train_and_predict()

st.subheader("🤖 Recommandation AI")

if prediction == 1 and confidence >= 0.60:
    cashout = round(1.7 + (confidence - 0.5), 2)
    st.success(f"✅ PLAY — Cashout conseillé : x{cashout}")
else:
    st.error("⛔ SKIP / WAIT")

st.write("🔍 Confidence :", round(confidence * 100, 2), "%")

# ===== Graph =====
st.subheader("📈 Historique Crash")

fig, ax = plt.subplots()
ax.plot(df["value"].tail(50))
ax.axhline(1.7, linestyle="--")
ax.set_ylabel("Multiplier")
st.pyplot(fig)

# ===== Stats =====
st.subheader("📊 Statistiques")

col1, col2, col3 = st.columns(3)
col1.metric("Dernière valeur", round(df["value"].iloc[-1], 2))
col2.metric("LOW streak", int(df["low_streak"].iloc[-1]))
col3.metric("Nombre de tours", len(df))
