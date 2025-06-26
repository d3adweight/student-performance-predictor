def encode_data(df):
    df_encoded = df.copy()
    df_encoded['Extracurricular Activities'] = df_encoded['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return df_encoded
