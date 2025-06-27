import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Constants
FACTOR_MAPPING = {
    "Previous Scores": "Nilai Sebelumnya",
    "Hours Studied": "Jam Belajar",
    "Sleep Hours": "Jam Tidur", 
    "Sample Question Papers Practiced": "Latihan Soal Ujian",
    "Extracurricular Activities": "Kegiatan Ekstrakurikuler"
}

def show_feature_correlation(df):
    corr = df.corr(numeric_only=True)['Performance Index'].drop('Performance Index')
    corr_sorted = corr.sort_values(key=lambda x: abs(x), ascending=False)

    df_korelasi = corr_sorted.reset_index().rename(columns={
        'index': 'Faktor',
        'Performance Index': 'Korelasi'
    })
    df_korelasi['Faktor'] = df_korelasi['Faktor'].map(FACTOR_MAPPING).fillna(df_korelasi['Faktor'])

    st.dataframe(df_korelasi, use_container_width=True)
    
def plot_predictions(df):
    X = df.drop(columns='Performance Index')
    y = df['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7, s=60, color="#4C72B0", edgecolor="white")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='y = x (perfect prediction)')

    ax.set_xlabel("Nilai Aktual", fontsize=11)
    ax.set_ylabel("Nilai Prediksi", fontsize=11)
    ax.set_title("Prediksi vs Nilai Aktual", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.text(0.05, 0.95, f"MSE = {mse:.2f}\nR² = {r2:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    st.pyplot(fig)
    
def plot_user_contribution(user_input, model, feature_names):
    user_values = user_input.values.flatten()
    coefs = model.coef_
    contributions = user_values * coefs

    mapped_features = [FACTOR_MAPPING.get(feat, feat) for feat in feature_names]

    df = pd.DataFrame({
        'Faktor': mapped_features,
        'Nilai Input': user_values,
        'Koefisien Model': coefs,
        'Kontribusi': contributions
    }).sort_values(by='Kontribusi', ascending=False)

    # Plot
    fig, ax = plt.subplots()
    ax.barh(df['Faktor'], df['Kontribusi'], color='salmon')
    ax.set_xlabel("Kontribusi terhadap Prediksi")
    ax.set_title("🔍 Faktor yang Mempengaruhi Prediksi Anda")
    ax.invert_yaxis()

    st.pyplot(fig)
