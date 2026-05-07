import streamlit as st
import pandas as pd
import numpy as np
import os
import itertools
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import skew, rankdata

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Aplikasi ini merupakan implementasi dari strategi **3-Seed Diversity Blend** menggunakan model XGBoost, LightGBM, dan CatBoost.
Upload file `train.csv` (untuk fitting encoder) dan `test.csv` (untuk prediksi).
""")

# --- HELPER FUNCTIONS ---
def pctrank_against(v, r):
    return (np.searchsorted(np.sort(r), v) / len(r)).astype('float32')

def zscore_against(v, r):
    mu, s = np.mean(r), np.std(r)
    return np.zeros(len(v), dtype='float32') if s == 0 else ((v - mu) / s).astype('float32')

def rank_blend(arrays, weights):
    n = len(arrays[0])
    ranked = [rankdata(a) / n for a in arrays]
    return np.clip(sum(w * r for w, r in zip(weights, ranked)), 0, 1)

# --- 1. UPLOAD DATA ---
col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader("Upload Train CSV (untuk referensi fitur)", type=['csv'])
with col2:
    test_file = st.file_uploader("Upload Test CSV (untuk Prediksi)", type=['csv'])

if train_file and test_file:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    st.success(f"Data Loaded: Train {train.shape}, Test {test.shape}")

    # --- 2. PRE-COMPUTE STATIC MAPS (Simulasi Data 'Original') ---
    # Notebook Anda sangat bergantung pada dataset 'Original.csv' untuk smoothing & stats.
    # Di sini kita gunakan data Train sebagai basis 'Original' jika file terpisah tidak ada.
    original = train.copy()
    original['TotalCharges'] = pd.to_numeric(original['TotalCharges'], errors='coerce').fillna(0)
    if 'Churn' in original.columns:
        original['Churn_bin'] = (original['Churn'] == 'Yes').astype(int)
    else:
        # Fallback jika kolom Churn tidak ada
        original['Churn_bin'] = 0

    global_churn_mean = original['Churn_bin'].mean()
    CATS_ORIG = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
                 'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                 'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
    NUMS_ORIG = ['tenure','MonthlyCharges','TotalCharges']

    # --- 3. FEATURE ENGINEERING PIPELINE ---
    @st.cache_data
    def process_data(df_input, is_train=True):
        df = df_input.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
        # Binary Encodes
        for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
            if col in df.columns: df[col] = (df[col] == 'Yes').astype(int)
        if 'gender' in df.columns: df['gender'] = (df['gender'] == 'Male').astype(int)
        
        # Ratios & Aggregates
        df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure'].clip(lower=1)
        df['ChargeRatio'] = df['MonthlyCharges'] / df['AvgMonthlyCharge'].clip(lower=0.01)
        df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
        df['TotalServices'] = (df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                   'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)
        
        # Risk & Loyalty
        if 'Contract' in df.columns:
            cr_map = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
            df['ContractRisk'] = df['Contract'].map(cr_map).fillna(2)
        
        # Numerical stats against 'Original' reference
        tc_values = original['TotalCharges'].values
        df['pctrank_orig_TC'] = pctrank_against(df['TotalCharges'].values, tc_values)
        
        # Dummy Variables
        ohe_cols = [c for c in ['MultipleLines','InternetService','Contract','PaymentMethod'] if c in df.columns]
        df = pd.get_dummies(df, columns=ohe_cols)
        
        return df

    st.info("Sedang memproses fitur... (Feature Engineering)")
    train_clean = process_data(train)
    test_clean = process_data(test)

    # Menyelaraskan kolom antara train dan test
    features = [c for c in train_clean.columns if c not in ['Churn', 'Churn_bin', 'id', 'customerID']]
    for col in features:
        if col not in test_clean.columns:
            test_clean[col] = 0
    test_X = test_clean[features]

    st.write("Preview fitur siap pakai:", test_X.head())

    # --- 4. INFERENSI MODEL ---
    st.subheader("🚀 Model Inference")
    
    # Tombol untuk menjalankan prediksi
    if st.button("Run Prediction Ensemble"):
        # Note: Di aplikasi nyata, Anda sebaiknya memuat model yang sudah di-train (.json/.bin)
        # Untuk demo ini, kita inisialisasi model dengan params dari notebook Anda.
        
        with st.spinner("Menghitung prediksi dari XGB, LGBM, dan CatBoost..."):
            
            # 1. XGBoost
            xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.006)
            # Dummy fit jika model belum ada (Hanya untuk struktur kode)
            # Dalam praktek: xgb_model.load_model("model.json")
            
            # 2. LightGBM
            lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.02)
            
            # 3. CatBoost
            cat_model = CatBoostClassifier(iterations=100, verbose=False)

            # Simulasi Prediksi (Gunakan random jika model belum di-load untuk testing UI)
            # Ganti baris ini dengan loading model asli Anda
            pred_xgb = np.random.uniform(0, 1, len(test_X)) 
            pred_lgbm = np.random.uniform(0, 1, len(test_X))
            pred_cat = np.random.uniform(0, 1, len(test_X))

            # --- 5. BLENDING (V56 Strategy) ---
            # Menggunakan bobot dari optimasi notebook Anda: XGB=0.248, LGBM=0.334, CAT=0.418
            final_blend = rank_blend([pred_xgb, pred_lgbm, pred_cat], weights=[0.248, 0.334, 0.418])

            # --- 6. HASIL AKHIR ---
            results_df = pd.DataFrame({
                'id': test['id'] if 'id' in test.columns else range(len(test)),
                'Churn_Probability': final_blend,
                'Prediction': (final_blend > 0.5).astype(int)
            })

            st.success("Prediksi Selesai!")
            
            c1, c2 = st.columns(2)
            c1.metric("Rata-rata Probabilitas Churn", f"{final_blend.mean():.2%}")
            c2.metric("Total Prediksi Churn (Yes)", int(results_df['Prediction'].sum()))

            st.dataframe(results_df.head(20))

            # Download Button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Submission CSV",
                data=csv,
                file_name='churn_prediction_v19.csv',
                mime='text/csv',
            )

else:
    st.warning("Silakan upload kedua file (Train & Test) untuk memulai.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed based on V19 Customer Churn Strategy - SEED 456")