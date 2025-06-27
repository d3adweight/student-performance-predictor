import streamlit as st
from data.load_data import load_dataset
from utils.forms import input_form
from utils.visualization import show_feature_correlation, plot_user_contribution, plot_predictions
from utils.preprocessing import encode_data
from models.linear_model import train_and_predict

def main():
    st.set_page_config(page_title="Prediksi Prestasi Pelajar", layout="centered")
    st.title("🎓 Prediksi Prestasi Akademik Pelajar")

    st.markdown("---")
    st.markdown("### 📌 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi *Performance Index* atau indeks prestasi akademik berdasarkan faktor-faktor seperti nilai sebelumnya, jam belajar, dan kebiasaan belajar.

    📁 _Sumber dataset diambil dari [Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)_.
    """)
    
    # Load dan encode data
    df = load_dataset("assets/Student_Performance.csv")
    df_encoded = encode_data(df)

    st.markdown("---")
    st.markdown("### ⚙️ Tentang Model")
    st.write("""
    Model menggunakan algoritma **Regresi Linear**, yang menghitung kontribusi setiap faktor terhadap prediksi nilai akhir.

    Faktor yang digunakan:
    - Nilai sebelumnya
    - Jam belajar
    - Jam tidur
    - Latihan soal
    - Kegiatan ekstrakurikuler
    """)

    st.markdown("---")
    st.subheader("📊 Korelasi Faktor terhadap Performa")
    st.caption("Hubungan antara tiap faktor dengan *Performance Index* berdasarkan data historis.")
    show_feature_correlation(df_encoded)

    st.markdown("---")
    st.subheader("📝 Formulir Input Data Pelajar")
    st.caption("Isi formulir berikut untuk memprediksi performa akademik Anda.")
    user_input = input_form()

    if user_input is not None:
        prediction, mse, r2, model = train_and_predict(df_encoded, user_input)

        # Kategorisasi
        q1 = df['Performance Index'].quantile(0.25)
        q3 = df['Performance Index'].quantile(0.75)

        if prediction > q3:
            kategori = "Tinggi"
            narasi = "Prestasi akademik tergolong **tinggi**. Pertahankan konsistensi dan terus eksplorasi materi lanjut."
        elif prediction >= q1:
            kategori = "Sedang"
            narasi = "Prestasi akademik tergolong **sedang**. Disarankan untuk meningkatkan jam belajar dan memperbanyak latihan soal."
        else:
            kategori = "Rendah"
            narasi = "Prestasi akademik tergolong **rendah**. Perlu perhatian serius. Coba atur ulang strategi belajar dan minta bantuan akademik jika perlu."

        # Output
        st.subheader("📊 Hasil Prediksi")
        st.metric("Prediksi Performance Index", f"{prediction:.2f}")

        if kategori == "Tinggi":
            st.success(f"Tingkat Prestasi: {kategori}")
        elif kategori == "Sedang":
            st.warning(f"Tingkat Prestasi: {kategori}")
        else:
            st.error(f"Tingkat Prestasi: {kategori}")

        st.markdown("---")
        st.markdown(f"""
        #### ℹ️ Penjelasan Kategori
        - **Tinggi**: > {q3:.0f}
        - **Sedang**: {q1:.0f} – {q3:.0f}
        - **Rendah**: < {q1:.0f}

        Kategori dibentuk berdasarkan distribusi kuartil dari data.
        """)

        st.markdown("---")
        st.subheader("🧠 Analisis")
        st.write(narasi)

        # Evaluasi dan visualisasi
        st.markdown("\n")
        with st.expander("📏 Evaluasi Model & Visualisasi"):
            st.write(f"- **Mean Squared Error (MSE)**: `{mse:.4f}`")
            st.write(f"- **R-squared (R²)**: `{r2:.4f}`")
            plot_predictions(df_encoded)
            plot_user_contribution(user_input, model, df_encoded.drop(columns='Performance Index').columns)

if __name__ == "__main__":
    main()
