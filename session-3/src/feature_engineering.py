def add_clinical_features(df):
    # Example: add simple ratios that can be clinically plausible
    df = df.copy()
    if 'mean_perimeter' in df.columns and 'mean_area' in df.columns:
        df['perimeter_area_ratio'] = df['mean_perimeter'] / (df['mean_area'] + 1e-6)
    return df
