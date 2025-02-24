import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# Tab Navigasi
tab1, tab2 = st.tabs(["🏠 Home", "📊 Prediksi"])

# Tab Home
with tab1:
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Harga Rumah")
    st.write("""
    Aplikasi ini menggunakan model Machine Learning (XGBoost + Optuna) untuk memprediksi harga rumah 
    berdasarkan fitur seperti jumlah kamar, luas tanah, daya listrik, dan lokasi kecamatan.
    
    **Cara Menggunakan:**
    1. Buka tab **Prediksi**.
    2. Isi semua input yang tersedia.
    3. Klik tombol **Prediksi Harga**.
    4. Harga estimasi akan ditampilkan dalam bentuk rupiah.
    
    🚀 **Mulai sekarang dengan membuka tab Prediksi!**
    """)

# Tab Prediksi
with tab2:
    st.title("📊 Prediksi Harga Rumah")

    # Layout dengan 2 kolom untuk input user
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏡 Informasi Properti")
        kamar_tidur = st.number_input("Jumlah Kamar Tidur:", min_value=1, max_value=10, value=3)
        kamar_mandi = st.number_input("Jumlah Kamar Mandi:", min_value=1, max_value=10, value=2)
        luas_tanah = st.number_input("Luas Tanah (m²):", min_value=10, max_value=1000, value=100)
        luas_bangunan = st.number_input("Luas Bangunan (m²):", min_value=10, max_value=1000, value=80)

    with col2:
        st.subheader("⚡ Spesifikasi Tambahan")
        daya_listrik = st.number_input("Daya Listrik (Watt):", min_value=450, max_value=6600, value=1300)
        jumlah_lantai = st.number_input("Jumlah Lantai:", min_value=1, max_value=5, value=1)
        carport = st.number_input("Carport (Jumlah Mobil):", min_value=0, max_value=5, value=1)
        kamar_tidur_pembantu = st.number_input("Kamar Tidur Pembantu:", min_value=0, max_value=5, value=0)
        kamar_mandi_pembantu = st.number_input("Kamar Mandi Pembantu:", min_value=0, max_value=5, value=0)

    # Pilihan Kecamatan
    kecamatan_list = [
        'Balaraja', 'Cikupa', 'Cisauk', 'Curug', 'Jatiuwung', 'Jayanti', 'Kadu',
        'Kelapa Dua', 'Kosambi', 'Kresek', 'Legok', 'Mauk', 'Pagedangan',
        'Panongan', 'Pasar Kemis', 'Rajeg', 'Sepatan', 'Sindang Jaya', 'Solear',
        'Teluk Naga', 'Tigaraksa'
    ]
    selected_kecamatan = st.selectbox("🏙️ Pilih Kecamatan:", kecamatan_list)

    # One-Hot Encoding Kecamatan
    kecamatan_encoded = {f'kec_{kec}': 0 for kec in kecamatan_list}
    kecamatan_encoded[f'kec_{selected_kecamatan}'] = 1  # Set kecamatan yang dipilih ke 1

    # Gabungkan input user ke DataFrame
    df_input = pd.DataFrame({
        'Kamar Tidur': [kamar_tidur],
        'Kamar Mandi': [kamar_mandi],
        'Luas Tanah': [np.log1p(luas_tanah)],  # Transformasi log1p
        'Luas Bangunan': [np.log1p(luas_bangunan)],  # Transformasi log1p
        'Daya Listrik': [daya_listrik],
        'Jumlah Lantai': [jumlah_lantai],
        'Carport': [carport],
        'Kamar Tidur Pembantu': [kamar_tidur_pembantu],
        'Kamar Mandi Pembantu': [kamar_mandi_pembantu]
    })

    df_final = pd.concat([df_input, pd.DataFrame([kecamatan_encoded])], axis=1)

    # Urutkan fitur agar sesuai dengan model
    expected_columns = [
        'Kamar Tidur', 'Kamar Mandi', 'Luas Tanah', 'Luas Bangunan', 'Daya Listrik',
        'Jumlah Lantai', 'Carport', 'Kamar Tidur Pembantu', 'Kamar Mandi Pembantu'
    ] + [f'kec_{kec}' for kec in kecamatan_list]

    df_final = df_final[expected_columns]

    # Tambahkan jarak antar elemen
    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol Prediksi di Tengah
    col1, col2, col3 = st.columns([1, 2, 1])  # Membuat layout 3 kolom (kolom tengah lebih besar)

    with col2:
        predict_button = st.button("🔍 Prediksi Harga", use_container_width=True)

    # Jika tombol ditekan
    if predict_button:
        try:
            # Load model
            with open("Model/xgboost_optuna.pkl", "rb") as f:
                model_xgb = pickle.load(f)

            # Prediksi harga rumah dalam bentuk log
            predicted_price_log = model_xgb.predict(df_final)

            # Konversi kembali ke harga asli
            predicted_price = np.expm1(predicted_price_log) 

            # Menampilkan hasil prediksi
            st.subheader("💰 Estimasi Harga Rumah:")
            st.success(f"Rp {predicted_price[0]:,.0f}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")