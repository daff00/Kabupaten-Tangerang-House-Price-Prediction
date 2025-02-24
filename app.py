import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# Tab Navigasi
tab1, tab2, tab3 = st.tabs(["🏠 Home", "⚠️Disclaimer", "📊 Prediksi"])

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
             
    *Notes: Perhatikan
    
    🚀 **Mulai sekarang dengan membuka tab Prediksi!**
    """)

# Tab Disclaimer
# Tab Disclaimer
with tab2:
    st.title("⚠️ Disclaimer")
    st.markdown("""
    **Harap Perhatian** 🏠🔍  

    Website ini menyediakan estimasi harga rumah berdasarkan model prediktif yang dikembangkan menggunakan teknik pembelajaran mesin.  

    ---
    
    ⚠️ **Hanya Sebagai Referensi**  
    Prediksi yang diberikan oleh sistem ini hanya bersifat estimasi dan **tidak dapat dijadikan acuan pasti** dalam transaksi jual beli properti.  

    📊 **Ketergantungan pada Data**  
    Akurasi prediksi bergantung pada data yang digunakan dalam pelatihan model. **Faktor eksternal** seperti kondisi ekonomi dan tren pasar **tidak selalu terakomodasi** dalam model ini.  

    🚫 **Tidak Menjamin Akurasi**  
    Meskipun model telah dioptimalkan, hasil prediksi bisa berbeda dengan harga pasar sebenarnya.  

    💡 **Bukan Saran Keuangan atau Properti**  
    Website ini **bukan merupakan saran investasi**. Disarankan untuk **berkonsultasi dengan agen properti** atau profesional terkait sebelum mengambil keputusan.  

    🔐 **Privasi Data**  
    Website ini **tidak menyimpan atau membagikan data** yang diinput pengguna tanpa izin.  

    ---
    
    Dengan menggunakan layanan ini, pengguna **menyetujui bahwa pengembang website tidak bertanggung jawab atas keputusan yang dibuat berdasarkan hasil prediksi**.
    """, unsafe_allow_html=True)

# Tab Prediksi
with tab3:
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
    # Kecamatan yang tersedia dalam model (harus sesuai dengan saat model dilatih)
    kecamatan_tersedia = [
        'Balaraja', 'Cikupa', 'Cisauk', 'Curug', 'Jatiuwung', 'Jayanti', 'Kadu',
        'Kelapa Dua', 'Kosambi', 'Kresek', 'Legok', 'Mauk', 'Pagedangan',
        'Panongan', 'Pasar Kemis', 'Rajeg', 'Sepatan', 'Sindang Jaya', 'Solear',
        'Teluk Naga', 'Tigaraksa'
    ]

    # Kecamatan yang tidak memiliki data harga rumah
    kecamatan_tidak_tersedia = [
        'Gunung Kaler', 'Jambe', 'Kemiri', 'Mekar Baru', 
        'Pakuhaji', 'Sepatan Timur', 'Sukadiri', 'Sukamulya'
    ]

    # Pilih kecamatan berdasarkan daftar yang tersedia
    selected_kecamatan = st.selectbox("🏙️ Pilih Kecamatan:", kecamatan_tersedia + kecamatan_tidak_tersedia)

    # Tombol Prediksi di Tengah
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        predict_button = st.button("🔍 Prediksi Harga", use_container_width=True)

    # Jika tombol ditekan
    if predict_button:
        if selected_kecamatan in kecamatan_tidak_tersedia:
            st.error(f"❌ Mohon Maaf! Data Harga Rumah untuk Kecamatan **{selected_kecamatan}** belum tersedia.")
        else:
            try:
                # One-Hot Encoding hanya untuk kecamatan yang tersedia
                kecamatan_encoded = {f'kec_{kec}': 0 for kec in kecamatan_tersedia}
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

                # Pastikan urutan fitur sesuai dengan yang digunakan saat model dilatih
                expected_columns = [
                    'Kamar Tidur', 'Kamar Mandi', 'Luas Tanah', 'Luas Bangunan', 'Daya Listrik',
                    'Jumlah Lantai', 'Carport', 'Kamar Tidur Pembantu', 'Kamar Mandi Pembantu'
                ] + [f'kec_{kec}' for kec in kecamatan_tersedia]

                df_final = df_final[expected_columns]

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