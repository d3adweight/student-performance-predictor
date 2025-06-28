import streamlit as st
from data.load_data import load_dataset
from utils.forms import input_form
from utils.visualization import show_correlation_and_coefficients, plot_user_contribution, plot_predictions
from utils.preprocessing import encode_data
from models.linear_model import train_and_predict

def main():
    st.set_page_config(page_title="Prediksi Tingkat Performa Pelajar", layout="centered")
    st.title("📚 Prediksi Tingkat Performa Akademik Pelajar")

    st.markdown("---")
    st.markdown("### 📌 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk membantu memprediksi **tingkat performa akademik pelajar** berdasarkan sejumlah faktor penting yang memengaruhi hasil belajar.

    Dengan memasukkan data seperti **nilai sebelumnya**, **jam belajar**, hingga **aktivitas ekstrakurikuler**, aplikasi ini akan menghitung dan menampilkan prediksi performa secara interaktif.

    📁 _Dataset bersumber dari [Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)_.
    """)

    # Load dan encode data
    df = load_dataset("assets/Student_Performance.csv")
    df_encoded = encode_data(df)

    st.markdown("---")
    st.markdown("### ⚙️ Tentang Model")
    st.write("""
    Aplikasi ini menggunakan algoritma **Regresi Linear**, yang mampu memetakan hubungan antara faktor-faktor belajar dengan hasil performa akhir pelajar.

    Faktor yang dipertimbangkan meliputi:
    - Nilai akademik sebelumnya
    - Jam belajar per minggu
    - Durasi tidur harian
    - Frekuensi latihan soal
    - Keterlibatan dalam kegiatan ekstrakurikuler

    Setiap faktor dianalisis untuk mengetahui seberapa besar pengaruhnya terhadap prediksi performa.
    """)

    st.markdown("---")
    show_correlation_and_coefficients(df_encoded)

    st.markdown("---")
    user_input = input_form()

    if user_input is not None:
        prediction, mse, r2, model = train_and_predict(df_encoded, user_input)

        # Kategorisasi
        q1 = df['Performance Index'].quantile(0.25)
        q3 = df['Performance Index'].quantile(0.75)

        if prediction > q3:
            kategori = "Tinggi"
            narasi = "Performa akademik Anda tergolong **tinggi**. Pertahankan semangat belajar dan terus kembangkan kemampuan!"
        elif prediction >= q1:
            kategori = "Sedang"
            narasi = "Performa akademik Anda berada di tingkat **sedang**. Perlu sedikit peningkatan fokus dan manajemen waktu belajar."
        else:
            kategori = "Rendah"
            narasi = "Performa akademik Anda tergolong **rendah**. Disarankan untuk meninjau kembali kebiasaan belajar dan mencari dukungan akademik jika diperlukan."

        # Output
        st.subheader("📈 Hasil Prediksi")
        st.metric("Performa Akademik Diprediksi", f"{prediction:.2f}")

        if kategori == "Tinggi":
            st.success(f"Tingkat Performa: {kategori}")
        elif kategori == "Sedang":
            st.warning(f"Tingkat Performa: {kategori}")
        else:
            st.error(f"Tingkat Performa: {kategori}")

        st.markdown("---")
        st.markdown(f"""
        #### ℹ️ Kategori Performa
        - **Tinggi**: > {q3:.0f}
        - **Sedang**: {q1:.0f} – {q3:.0f}
        - **Rendah**: < {q1:.0f}

        Kategori ini didasarkan pada distribusi kuartil dari data aktual pelajar.
        """)

        st.markdown("---")
        st.subheader("🧠 Analisis Hasil")
        st.write(narasi)

        st.markdown("\n")
        with st.expander("📏 Evaluasi Model & Visualisasi Tambahan"):
            st.write(f"- **Mean Squared Error (MSE)**: `{mse:.4f}`")
            st.write(f"- **R-squared (R²)**: `{r2:.4f}`")
            plot_predictions(df_encoded)
            plot_user_contribution(user_input, model, df_encoded.drop(columns='Performance Index').columns)

if __name__ == "__main__":
    main()
