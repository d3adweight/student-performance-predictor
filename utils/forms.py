import streamlit as st
import pandas as pd

def input_form():
    with st.form("student_form"):
        hours = st.slider("Jam Belajar per Hari", 0, 10, 5)
        prev_score = st.slider("Nilai Sebelumnya (0–100)", 0, 100, 70)
        ec = st.selectbox("Aktivitas Ekstrakurikuler", ["Yes", "No"])
        sleep = st.slider("Jam Tidur", 0, 12, 7)
        practice = st.slider("Jumlah Latihan Soal", 0, 10, 5)

        submit = st.form_submit_button("Prediksi")

        if submit:
            data = {
                "Hours Studied": hours,
                "Previous Scores": prev_score,
                "Extracurricular Activities": 1 if ec == "Yes" else 0,
                "Sleep Hours": sleep,
                "Sample Question Papers Practiced": practice
            }
            return pd.DataFrame([data])
    return None
