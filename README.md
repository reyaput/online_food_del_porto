# 🍔 Online Food Delivery Customer Churn Prediction System (OrderKu)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Business Understanding](#-business-understanding)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Data Understanding](#-data-understanding)
- [Key Insights (EDA)](#-key-insights-eda)
- [Machine Learning Model](#-machine-learning-model)
- [Business Recommendations](#-business-recommendations)

---

## 📋 Project Overview
Proyek ini bertujuan untuk membangun portofolio **Data Analysis & End-to-End Machine Learning** dalam industri *Online Food Delivery* (**OrderKu**). Sistem ini dirancang untuk memprediksi pelanggan yang berisiko berhenti memesan (*Churn* / `Output = No`) berdasarkan profil demografis, latar belakang ekonomi, dan umpan balik (*Feedback*) yang diberikan.

Dengan memadukan eksplorasi data mendalam (*EDA*), pencegahan kebocoran data (*data leakage prevention*), *threshold tuning* dengan batasan bisnis (*business constraint*), serta antarmuka interaktif berbasis **Streamlit**, tim operasional dan CRM dapat melakukan intervensi retensi secara cepat, terukur, dan efisien sebelum pelanggan beralih ke platform kompetitor.

---

## 💼 Business Understanding

### Problem Statement
Dalam industri pesan-antar makanan daring (*food delivery*), kompetisi sangat ketat dengan biaya akuisisi pelanggan (*Customer Acquisition Cost* / CAC) yang tinggi akibat perang promo dan diskon. 
* Dataset menunjukkan tingkat *Churn* sebesar **~22.4%** (hampir 1 dari 4 pelanggan berhenti memesan).
* Mengakuisisi pelanggan baru membutuhkan biaya **5x hingga 7x lipat** dibandingkan mempertahankan pelanggan lama.
* Pelanggan yang berhenti memesan menurunkan nilai transaksi bruto (*Gross Merchandise Value* / GMV) serta merugikan ekosistem merchant dan mitra pengemudi.

### Objectives
1. **Identifikasi Faktor Risiko (Root Cause Analysis):** Mengungkap pola demografis, sosio-ekonomi, dan sinyal kepuasan yang mendorong pelanggan berhenti menggunakan layanan.
2. **Pencegahan Data Leakage & Pemodelan Robust:** Merancang pipeline Machine Learning yang bebas dari kebocoran fitur dengan fokus evaluasi pada deteksi kelas minoritas (*Churn Recall & Churn Precision*).
3. **Threshold Tuning dengan Batasan Bisnis:** Mengoptimalkan *decision threshold* dari probabilitas prediksi dengan syarat *Precision Churn* $\ge$ 40%, sehingga tim retensi tidak membuang anggaran promo ke pelanggan yang sebenarnya loyal.
4. **Dashboard Operasional Interaktif:** Menyediakan aplikasi web (Streamlit) yang mendukung pengecekan risiko pelanggan satuan (*single prediction*) maupun pemrosesan massal (*batch CSV upload*) untuk kebutuhan tim CRM.

---

## 🛠️ Installation & Setup
Proyek ini dibangun menggunakan bahasa pemrograman Python. Disarankan menggunakan **Conda** atau **Virtualenv** untuk manajemen environment agar dependensi terisolasi dengan rapi.

### 1. Prerequisite
Pastikan Anda telah menginstal [Anaconda](https://www.anaconda.com/) / [Miniconda](https://docs.conda.io/en/latest/miniconda.html) atau Python 3.11+.

### 2. Setup Environment
Buka terminal (Anaconda Prompt / CMD / PowerShell) dan jalankan perintah berikut:

```bash
# 1. Buat environment baru bernama 'food_churn' dengan Python 3.11
conda create -n food_churn python=3.11 -y

# 2. Aktifkan environment
conda activate food_churn

# 3. Masuk ke direktori project (sesuaikan path folder Anda)
cd "path/to/online_food_del_porto"
```

### 3. Install Dependencies
Install library yang dibutuhkan menggunakan pip:

```bash
pip install -r requirements.txt
```

> File `requirements.txt` mencakup dependensi utama: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, dan `joblib`.

---

## 🚀 How to Run
Proyek ini menyediakan alur kerja yang mudah dijalankan, mulai dari pelatihan ulang model (*training pipeline*) hingga deployment aplikasi web (*Streamlit dashboard*).

### A. Melatih Ulang Model (Training Pipeline)
Jalankan skrip ini jika ingin melatih ulang model, memvalidasi cross-validation, mencari threshold optimal, atau memperbarui artefak metadata:

```bash
python train_model.py
```

* **Proses yang dijalankan:**
  1. Membaca dataset mentah (`data/raw/online_food_delivery_dataset.csv`).
  2. Melakukan data cleaning & menyimpan data perantara (`data/interim/df_interim.csv` & `data/processed/df_model_ready.csv`).
  3. Melakukan 5-Fold Stratified Cross-Validation pada Logistic Regression dan Random Forest.
  4. Melakukan OOF (*Out-Of-Fold*) threshold tuning dengan constraint bisnis (*Precision Churn* $\ge$ 40%).
  5. Menguji performa final pada Holdout Test Set (20%).
  6. Menyimpan model pipeline (`best_model.pkl`), feature names (`feature_names.pkl`), dan rekam jejak eksperimen lengkap (`model_metadata.json`) ke folder `models/`.

### B. Menjalankan Dashboard Prediksi (Streamlit App)
Jalankan perintah ini untuk membuka antarmuka aplikasi prediksi berbasis web:

```bash
streamlit run app.py
```

* **Akses Aplikasi:** Browser akan terbuka secara otomatis di `http://localhost:8501`.
* **Fitur Utama Aplikasi:**
  * **🧍 Input Manual:** Masukkan parameter profil pelanggan satu per satu untuk mendapatkan skor loyalitas (%), probabilitas churn (%), dan label status retensi secara real-time.
  * **📂 Upload CSV (Batch):** Unggah file CSV pelanggan secara massal, lengkapi dengan template unduhan, dan ekspor hasil prediksi beserta rekomendasi prioritas retensi.
  * **📊 Penjelasan Model:** Visualisasi transparansi model mencakup metrik evaluasi CV & Test, threshold keputusan, serta grafik *Top 10 Feature Importance*.

---

## 📊 Data Understanding
Dataset yang digunakan mencakup data survei pelanggan layanan pesan-antar makanan daring dengan **388 baris** dan **14 fitur**.

### Kamus Data (Feature Dictionary)
| Fitur | Deskripsi | Tipe Data | Status dalam Model |
| :--- | :--- | :--- | :--- |
| **Age** | Usia pelanggan (rentang 18 - 33 tahun) | Numerik | Digunakan (Input) |
| **Gender** | Jenis kelamin (`Male`, `Female`) | Kategorikal | Digunakan (Input) |
| **Marital Status** | Status pernikahan (`Single`, `Married`, `Prefer not to say`) | Kategorikal | Digunakan (Input) |
| **Occupation** | Status pekerjaan (`Student`, `Employee`, `Self Employeed`, `House wife`) | Kategorikal | Digunakan (Input) |
| **Monthly Income** | Rentang pendapatan bulanan (`No Income`, `Below Rs.10000`, `10001 to 25000`, `25001 to 50000`, `More than 50000`) | Kategorikal | Digunakan (Input) |
| **Educational Qualifications** | Tingkat pendidikan formal (`School`, `Graduate`, `Post Graduate`, `Ph.D`, `Uneducated`) | Kategorikal | Digunakan (Input) |
| **Feedback** | Sentimen ulasan terakhir yang diberikan pelanggan (`Positive`, `Negative`) | Kategorikal | Digunakan (Input) |
| **Output** | Status pemesanan kembali (`Yes` = Loyal/Stay, `No` = Churn) | Target | **Target Prediksi** |
| **Family size** | Jumlah anggota keluarga | Numerik | *Excluded* (Penyebab data leakage terhadap customer profile) |
| **Customer Type** | Tipe pelanggan | Kategorikal | *Excluded* (Beririsan langsung dengan definisi label) |
| **latitude** | Koordinat lintang tempat tinggal | Numerik | *Excluded* (Tidak relevan untuk profil retensi umum) |
| **longitude** | Koordinat bujur tempat tinggal | Numerik | *Excluded* (Tidak relevan untuk profil retensi umum) |
| **Pin code** | Kode pos area pengiriman | Numerik | *Excluded* (Kardinalitas lokasi lokal) |

---

## 📊 Key Insights (EDA)

Berdasarkan analisis eksplorasi data terhadap 388 pelanggan, berikut adalah temuan utama:

### 1. Tingkat Retensi & Churn Keseluruhan
* **Distribusi Target:** Sebanyak **77.6% (301 pelanggan)** berstatus Loyal (`Output = Yes`), sedangkan **22.4% (87 pelanggan)** berstatus Churn (`Output = No`).
* Meskipun kelas mayoritas adalah pelanggan setia, tingkat churn sebesar 22.4% menunjukkan bahwa lebih dari 1 dari 5 pelanggan berhenti menggunakan layanan.

### 2. Sinyal Ulasan (Feedback) Adalah Indikator Terkuat
* Pelanggan yang memberikan **Feedback "Negative"** memiliki tingkat Churn fantastis sebesar **74.6%**.
* Sebaliknya, pelanggan dengan **Feedback "Positive"** mencatat tingkat retensi/loyalitas sebesar **89.3%** (tingkat churn hanya 10.7%).
* *Insight:* Ulasan negatif bukan sekadar komplain, melainkan sinyal langsung bahwa pelanggan berada di ambang churn.

### 3. Dinamika Pekerjaan & Penghasilan Bulanan
* **Mahasiswa & Tanpa Penghasilan Sangat Loyal:** Kelompok `Student` (tingkat loyalitas **88.9%**) dan `No Income` (loyalitas **87.7%**) merupakan segmen pengguna paling stabil, umumnya mengandalkan layanan untuk kepraktisan makan harian.
* **Pekerja & Profesional Lebih Rentan Churn:** Kelompok `Employee` (churn **35.6%**) dan `Self Employeed` (churn **37.0%**), terutama pada rentang penghasilan `25001 to 50000` (churn **39.1%**), memiliki tingkat churn tertinggi.
* *Insight:* Kelompok berpenghasilan memiliki ekspektasi layanan lebih tinggi (kecepatan antar, kualitas kemasan) serta sensitif terhadap pengalaman yang mengecewakan.

### 4. Pengaruh Status Pernikahan & Usia
* **Status Pernikahan:** Pelanggan yang berstatus `Married` memiliki churn rate sebesar **38.9%**, jauh lebih tinggi dibandingkan pelanggan `Single` yang hanya mencatat churn rate **14.6%**.
* **Usia:** Pelanggan yang Churn memiliki rata-rata usia yang sedikit lebih tua (**26.0 tahun**) dibandingkan pelanggan Loyal (**24.2 tahun**). Pelanggan berkeluarga cenderung beralih memasak di rumah jika layanan pesan-antar tidak memenuhi standar higienis atau ketepatan waktu.

---

## 🤖 Machine Learning Model

### 1. Pendekatan Pemodelan & Pencegahan Data Leakage
* Menggunakan **`sklearn.pipeline.Pipeline`** yang mengintegrasikan `ColumnTransformer` (One-Hot Encoding untuk fitur kategorikal) langsung ke dalam estimator, memastikan transformasi fit dilakukan hanya pada fold training.
* Menetapkan **Churn (`Output = No` / Target = 0)** secara eksplisit sebagai *Positive Class* dalam kalkulasi metrik evaluasi (`pos_label=0`), karena fokus bisnis adalah mendeteksi pelanggan yang akan pergi.

### 2. Algoritma yang Dibandingkan (5-Fold Stratified CV)
| Model | CV Recall Churn | CV F1 Churn | CV ROC-AUC | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** *(Best Model)* | **70.0%** | **59.5%** | **0.839** | 77.4% |
| **Logistic Regression** | 64.3% | 59.1% | 0.824 | 79.4% |

* **Model Terpilih: Random Forest Classifier** karena memiliki kemampuan mendeteksi pelanggan churn yang lebih tinggi (*CV Recall Churn* 70.0% vs 64.3%) serta pemisahan probabilitas yang lebih unggul (*ROC-AUC* 0.839).

### 3. OOF Threshold Tuning dengan Business Constraint
Secara default, model klasifikasi menggunakan threshold probabilitas `0.50`. Untuk menyeimbangkan biaya penanganan retensi:
* Diterapkan aturan bisnis: **Precision Churn harus $\ge$ 40%** agar program retensi tidak menjangkau terlalu banyak *false alarm*.
* Berdasarkan evaluasi Out-Of-Fold (OOF), diperoleh **Threshold Optimal = 0.56 (56%)**.

### 4. Performa pada Holdout Test Set (Unseen Data - 20%)
| Metrik Evaluasi | Nilai pada Test Set | Interpretasi Bisnis |
| :--- | :---: | :--- |
| **Accuracy** | **84.6%** | Akurasi prediksi keseluruhan pada data pengujian. |
| **Churn Recall** | **76.5%** | Berhasil mendeteksi **13 dari 17** pelanggan yang sebenarnya churn. |
| **Churn Precision** | **61.9%** | Dari seluruh pelanggan yang diprediksi churn, **~62%** terbukti benar-benar churn. |
| **Churn F1-Score** | **68.4%** | Keseimbangan harmonis antara Recall dan Precision untuk kelas Churn. |
| **ROC-AUC** | **0.834** | Daya pembeda model yang sangat baik dalam memisahkan pelanggan loyal vs churn. |
| **Loyal F1-Score** | **89.8%** | Performa sangat tinggi dalam mempertahankan klasifikasi pelanggan setia. |

---

## 💡 Business Recommendations

Berdasarkan temuan EDA dan kapabilitas model prediksi, berikut langkah strategis yang direkomendasikan untuk manajemen **OrderKu**:

1. **Sistem Deteksi Dini Ulasan Negatif (*Closed-Loop Feedback System*):**
   * Karena pelanggan dengan ulasan negatif memiliki tingkat churn **74.6%**, terapkan SLA resolusi komplain maksimal **< 2 jam** untuk ulasan bertanda negatif.
   * Berikan voucher kompensasi instan atau pembebasan biaya kirim pada pesanan berikutnya bagi pelanggan yang komplain.

2. **Program Loyalitas Terarah untuk Segmen Pekerja & Pasangan Menikah:**
   * Segmen pekerja berpenghasilan Rs. 25k–50k dan pasangan menikah memiliki tingkat churn mendekati **40%**.
   * Rancang penawaran spesifik seperti *Family Meal Deals*, langganan bebas ongkir bulanan (*Pass Delivery*), dan garansi waktu sampai pada jam makan siang kantor atau makan malam keluarga.

3. **Optimasi Alokasi Anggaran Retensi dengan Model ML:**
   * Jangan menyebar promo diskon ke semua pelanggan secara acak (*spray and pray*).
   * Gunakan threshold **0.56**: prioritaskan intervensi diskon retensi hanya pada pelanggan yang memiliki skor probabilitas churn $\ge 56\%$. Hal ini menjaga margin profit perusahaan tetap sehat.

4. **Pertahankan Segmen Mahasiswa dengan Program Gamifikasi Hemat:**
   * Segmen mahasiswa merupakan basis pelanggan setia terbesar (**88.9% loyal**).
   * Pertahankan loyalitas mereka melalui paket hemat tanggal tua, program rujukan teman (*referral bonus*), dan poin reward loyalitas yang dapat ditukarkan dengan snack/minuman gratis.

---

### 👤 Author
* **Reynanda Arya Putra**
* Repository: [online_food_del_porto](https://github.com/reyaput/online_food_del_porto)
