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
- [Limitations](#-limitations)

---

## 📋 Project Overview

Proyek ini bertujuan untuk membangun portofolio **Data Analysis & End-to-End Machine Learning** dalam industri *Online Food Delivery* (**OrderKu**). Sistem dirancang untuk memprediksi pelanggan yang berisiko berhenti memesan (*Churn* / `Output = No`) berdasarkan profil demografis, latar belakang ekonomi, dan umpan balik (*Feedback*) yang tersedia pada data.

Dengan memadukan eksplorasi data (*EDA*), pencegahan potensi *data leakage*, *threshold tuning* dengan batasan bisnis (*business constraint*), serta antarmuka interaktif berbasis **Streamlit**, sistem dapat digunakan sebagai alat bantu prioritisasi retensi untuk tim operasional dan CRM.

> **Catatan metodologi:** model ini digunakan sebagai **decision-support / prioritization tool**, bukan sebagai bukti bahwa suatu faktor merupakan penyebab (*causal driver*) churn.

---

## 💼 Business Understanding

### Problem Statement

Dalam industri pesan-antar makanan daring (*food delivery*), kompetisi sangat ketat dengan biaya akuisisi pelanggan (*Customer Acquisition Cost* / CAC) yang tinggi akibat persaingan promo dan diskon.

- Dataset menunjukkan tingkat *churn* sebesar **~22.4%** (87 dari 388 pelanggan).
- Pelanggan yang berhenti memesan berpotensi menurunkan nilai transaksi bruto (*Gross Merchandise Value* / GMV) serta memengaruhi ekosistem merchant dan mitra pengemudi.
- Retensi pelanggan menjadi penting karena mempertahankan pelanggan yang sudah ada umumnya menjadi bagian penting dari strategi *customer lifecycle management*.

### Objectives

1. **Identifikasi Faktor Risiko:** Mengidentifikasi pola demografis, sosio-ekonomi, dan *feedback* yang berasosiasi dengan status churn.
2. **Pencegahan Data Leakage & Pemodelan Robust:** Merancang pipeline Machine Learning yang meminimalkan risiko kebocoran informasi dengan evaluasi yang berfokus pada kelas Churn.
3. **Threshold Tuning dengan Batasan Bisnis:** Mengoptimalkan *decision threshold* dari skor/probabilitas model dengan syarat *Precision Churn* `≥ 40%`, sehingga tim retensi dapat memfokuskan intervensi pada customer yang lebih berisiko.
4. **Dashboard Operasional Interaktif:** Menyediakan aplikasi web (Streamlit) yang mendukung pengecekan risiko pelanggan satuan (*single prediction*) maupun pemrosesan massal (*batch CSV upload*).

---

## 🛠️ Installation & Setup

Proyek ini dibangun menggunakan bahasa pemrograman Python. Disarankan menggunakan **Conda** atau **Virtualenv** untuk manajemen environment agar dependensi terisolasi dengan rapi.

### 1. Prerequisite

Pastikan Anda telah menginstal [Anaconda](https://www.anaconda.com/) / [Miniconda](https://docs.conda.io/en/latest/miniconda.html) dan menggunakan **Python 3.11**.

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

#### Proses yang dijalankan

1. Membaca dataset mentah (`data/raw/online_food_delivery_dataset.csv`).
2. Melakukan data cleaning & menyimpan data perantara (`data/interim/df_interim.csv` & `data/processed/df_model_ready.csv`).
3. Melakukan **5-Fold Stratified Cross-Validation** pada Logistic Regression dan Random Forest.
4. Menghasilkan **Out-Of-Fold (OOF)** prediction untuk threshold tuning dengan constraint bisnis (*Precision Churn* `≥ 40%`).
5. Menguji performa final pada **Holdout Test Set (20%)**.
6. Menyimpan model pipeline (`best_model.pkl`), feature names (`feature_names.pkl`), dan rekam jejak eksperimen (`model_metadata.json`) ke folder `models/`.

### B. Menjalankan Dashboard Prediksi (Streamlit App)

Jalankan perintah berikut:

```bash
streamlit run app.py
```

- **Akses Aplikasi:** Browser akan terbuka secara otomatis di `http://localhost:8501`.
- **🧍 Input Manual:** Masukkan parameter profil pelanggan satu per satu untuk mendapatkan skor model, probabilitas churn, dan label status retensi secara real-time.
- **📂 Upload CSV (Batch):** Unggah file CSV pelanggan secara massal, gunakan template unduhan, dan ekspor hasil prediksi beserta rekomendasi prioritas retensi.
- **📊 Penjelasan Model:** Menampilkan metrik evaluasi CV & Test, threshold keputusan, serta grafik *Top 10 Feature Importance*.

---

## 📊 Data Understanding

Dataset yang digunakan mencakup data survei pelanggan layanan pesan-antar makanan daring dengan **388 baris** dan **13 fitur**.

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
| **Family size** | Jumlah anggota keluarga | Numerik | *Excluded* (tidak digunakan dalam model berdasarkan keputusan analisis) |
| **Customer Type** | Tipe pelanggan | Kategorikal | *Excluded* (beririsan dengan informasi status target) |
| **latitude** | Koordinat lintang tempat tinggal | Numerik | *Excluded* (tidak digunakan karena pertimbangan relevansi model) |
| **longitude** | Koordinat bujur tempat tinggal | Numerik | *Excluded* (tidak digunakan karena pertimbangan relevansi model) |
| **Pin code** | Kode pos area pengiriman | Numerik | *Excluded* (kardinalitas lokasi relatif tinggi) |

> **Pengecekan temporal:** `Feedback` digunakan sebagai fitur prediktor. Agar penggunaan fitur ini valid untuk prediksi churn, `Feedback` harus tersedia pada titik waktu yang sama atau sebelum waktu prediksi. Karena dataset yang digunakan berbasis survei, hubungan temporal tersebut perlu diperhatikan ketika menggeneralisasikan model ke data produksi.

---

## 📊 Key Insights (EDA)

Berdasarkan analisis eksplorasi data terhadap 388 pelanggan, berikut adalah temuan utama:

### 1. Tingkat Retensi & Churn Keseluruhan

- **Distribusi Target:** Sebanyak **77.6% (301 pelanggan)** berstatus Loyal (`Output = Yes`), sedangkan **22.4% (87 pelanggan)** berstatus Churn (`Output = No`).
- Tingkat churn sebesar 22.4% menunjukkan bahwa lebih dari 1 dari 5 pelanggan dalam dataset termasuk kelas Churn.

### 2. Sinyal Ulasan (Feedback) Adalah Indikator Terkuat

- Pelanggan yang memberikan **Feedback "Negative"** memiliki tingkat Churn sebesar **74.6%**.
- Sebaliknya, pelanggan dengan **Feedback "Positive"** mencatat tingkat retensi/loyalitas sebesar **89.3%** (tingkat churn 10.7%).
- **Insight:** `Feedback` memiliki asosiasi yang kuat dengan status churn pada dataset ini dan dapat dipertimbangkan sebagai sinyal risiko retensi.

> Hindari menyebut `Feedback` sebagai **penyebab langsung** churn karena analisis ini bersifat observasional dan tidak menguji hubungan sebab-akibat.

### 3. Dinamika Pekerjaan & Penghasilan Bulanan

- **Mahasiswa & Tanpa Penghasilan:** Kelompok `Student` memiliki tingkat loyalitas **88.9%**, sedangkan kelompok `No Income` memiliki loyalitas **87.7%**.
- **Pekerja & Profesional:** Kelompok `Employee` memiliki churn **35.6%** dan `Self Employeed` **37.0%**. Rentang penghasilan `25001 to 50000` memiliki churn **39.1%**.
- **Insight:** Terdapat perbedaan tingkat churn antar kelompok pekerjaan dan rentang pendapatan. Namun, temuan ini menunjukkan **asosiasi**, bukan bukti bahwa pekerjaan atau pendapatan menyebabkan churn.

### 4. Pengaruh Status Pernikahan & Usia

- **Status Pernikahan:** Customer berstatus `Married` memiliki churn rate **38.9%**, sedangkan customer `Single` memiliki churn rate **14.6%**.
- **Usia:** Customer Churn memiliki rata-rata usia sedikit lebih tua (**26.0 tahun**) dibandingkan customer Loyal (**24.2 tahun**).
- **Insight:** Dataset menunjukkan adanya perbedaan pola retensi berdasarkan status pernikahan dan usia, tetapi tidak dapat digunakan untuk menyimpulkan penyebab perilaku tersebut.

---

## 🤖 Machine Learning Model

### 1. Target & Preprocessing

Target asli:

```text
Target = 1 -> Loyal  (Output = Yes)
Target = 0 -> Churn  (Output = No)
```

Untuk evaluasi bisnis, **Churn (`Target = 0`) diperlakukan sebagai positive class** menggunakan `pos_label=0`.

Model menggunakan:

- `ColumnTransformer`
- `OneHotEncoder` untuk fitur kategorikal
- `Pipeline` untuk menggabungkan preprocessing dan estimator

Preprocessing berada di dalam pipeline sehingga proses transformasi tetap konsisten antara training, cross-validation, dan inference.

### 2. Algoritma yang Dibandingkan

Dua algoritma dibandingkan menggunakan **5-Fold Stratified Cross-Validation (CV)**:

- Logistic Regression
- Random Forest Classifier

**Hasil CV:**

| Model | Churn Recall | Churn Precision | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| **Random Forest** | **70.0% ± 8.3%** | 56.6% ± 21.8% | **59.5% ± 6.8%** | **0.839 ± 0.057** |
| Logistic Regression | 64.3% ± 6.4% | **57.3% ± 16.0%** | 59.1% ± 6.1% | 0.824 ± 0.053 |

Random Forest dipilih sebagai model final karena memiliki **CV Churn Recall, CV Churn F1, dan CV ROC-AUC** yang sedikit lebih tinggi dibandingkan Logistic Regression.

> **Catatan:** Perbedaan performa CV cukup tipis, sehingga pemilihan Random Forest sebaiknya dipahami sebagai keputusan berdasarkan keseluruhan hasil evaluasi dan tujuan bisnis, bukan karena perbedaannya sangat besar.

### 3. OOF Threshold Tuning dengan Business Constraint

Selain threshold default `0.50`, project ini melakukan **Out-of-Fold (OOF) threshold tuning** pada training set.

#### Constraint bisnis

```text
Precision Churn >= 40%
```

Tujuannya adalah mencegah threshold terlalu rendah sehingga terlalu banyak customer Loyal salah ditandai sebagai Churn.

**Hasil tuning OOF:**

| Model | Threshold | Recall Churn | Precision Churn | F1 Churn | Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.63 | 61.4% | **67.2%** | **64.2%** | **84.5%** |
| **Random Forest** | **0.56** | **65.7%** | 61.3% | 63.4% | 82.9% |

Threshold final untuk Random Forest:

```text
0.56 (56%)
```

Threshold dipilih menggunakan **training-set OOF prediction**. **Holdout test set tidak digunakan untuk memilih threshold.**

> **Interpretasi threshold:** `0.56` adalah batas keputusan model untuk mengategorikan customer sebagai prioritas Churn. Nilai ini sebaiknya disebut sebagai **model score/probability threshold**, bukan sebagai kepastian bahwa customer memiliki peluang churn tepat 56%.

### 4. Final Holdout Test Set

Data dibagi menjadi:

```text
Train = 310 customer
Test  = 78 customer
```

Test set disimpan sebagai data yang tidak digunakan dalam pemilihan model maupun threshold.

Dengan **Random Forest + threshold Churn 0.56**, hasil pada holdout test set:

| Metrik Evaluasi | Nilai | Interpretasi |
|---|---:|---|
| **Accuracy** | **84.6%** | Proporsi prediksi benar pada keseluruhan test set. |
| **Churn Recall** | **76.5%** | Berhasil mendeteksi **13 dari 17** customer yang sebenarnya Churn. |
| **Churn Precision** | **61.9%** | Sekitar 62% customer yang ditandai Churn memang merupakan Churn. |
| **Churn F1-Score** | **68.4%** | Keseimbangan precision dan recall untuk kelas Churn. |
| **Recall Loyal** | **86.9%** | Sebagian besar customer Loyal berhasil dikenali sebagai Loyal. |
| **Loyal F1-Score** | **89.8%** | Performa klasifikasi kelas Loyal. |
| **ROC-AUC** | **0.834** | Kemampuan model membedakan kelas berdasarkan ranking skor/probabilitas. |
| **Average Precision (Churn)** | **0.722** | Ringkasan performa precision-recall untuk kelas Churn. |

### 5. Confusion Matrix

```text
             Predicted
             Churn  Loyal
Actual Churn    13     4
Actual Loyal     8    53
```

Interpretasi:

- **13 dari 17** customer Churn berhasil terdeteksi.
- **4 dari 17** customer Churn terlewat.
- **53 dari 61** customer Loyal berhasil dikenali sebagai Loyal.
- **8 dari 61** customer Loyal salah ditandai sebagai Churn.

### 6. Interpretasi Bisnis

Hasil model menunjukkan trade-off yang cukup seimbang pada holdout test set.

**Churn Recall 76.5%** berarti sebagian besar customer yang benar-benar Churn berhasil masuk ke daftar prioritas retensi.

**Churn Precision 61.9%** berarti mayoritas customer yang ditandai sebagai Churn memang berasal dari kelas Churn, sehingga model tidak sekadar menandai hampir semua customer sebagai berisiko.

Dengan demikian, model lebih cocok digunakan sebagai **decision-support / prioritization tool** untuk tim CRM, bukan sebagai keputusan otomatis bahwa seorang customer pasti akan Churn.

---

## 💡 Business Recommendations

### 1. Gunakan Model sebagai Prioritas Retensi

Gunakan skor churn dari model untuk membuat daftar prioritas pelanggan yang perlu ditinjau oleh tim CRM.

Dengan threshold **0.56**, customer dengan **model score/probability Churn ≥ 0.56** masuk kelompok prioritas retensi.

### 2. Hindari Pendekatan “Spray and Pray”

Model membantu membatasi intervensi agar tidak diberikan secara merata kepada seluruh customer.

Tujuannya adalah mengarahkan anggaran retensi kepada customer yang memiliki risiko Churn lebih tinggi.

### 3. Gunakan Feedback sebagai Sinyal Operasional

`Feedback` merupakan salah satu fitur model. Feedback dapat digunakan sebagai konteks tambahan ketika tim CRM menentukan jenis tindak lanjut yang sesuai.

Contohnya, customer dengan pengalaman negatif dapat diarahkan ke proses recovery atau penanganan keluhan.

> Penggunaan `Feedback` untuk prediksi produksi perlu memperhatikan apakah informasi tersebut sudah tersedia sebelum waktu intervensi retensi dilakukan.

### 4. Validasi dengan Data Produksi

Sebelum digunakan untuk otomatisasi kampanye, model sebaiknya divalidasi menggunakan data transaksi nyata seperti:

- frekuensi order,
- recency transaksi,
- monetary value,
- jumlah order sebelumnya,
- histori penggunaan promo,
- histori komplain.

Data transaksi tersebut dapat membantu model menggambarkan perilaku customer secara lebih aktual dibandingkan data survey saja.

---

## ⚠️ Limitations

1. **Dataset kecil**  
   Dataset terdiri dari 388 observasi dan test set 78 observasi. Metric pada satu split dapat berubah apabila pembagian data berubah.

2. **Data survey**  
   Dataset berbasis survey, bukan data transaksi produksi. Model menunjukkan kemampuan prediksi pada dataset ini dan **belum membuktikan hubungan sebab-akibat**.

3. **Ketersediaan Feedback**  
   Model memakai `Feedback` sebagai salah satu fitur. Penggunaan fitur ini terbatas pada kondisi ketika feedback memang tersedia sebelum waktu prediksi/intervensi.

4. **Threshold bukan kepastian Churn**  
   Threshold `0.56` adalah batas keputusan model untuk prioritas retensi, bukan berarti customer pasti akan Churn.

5. **Duplicate rows**  
   Sebanyak **103 duplicate rows** terdeteksi dan tidak di-drop sesuai keputusan analisis dataset. Pada data produksi dengan customer ID unik, pemeriksaan duplicate sebaiknya dilakukan kembali.

6. **Generalisasi ke data produksi**  
   Hasil evaluasi berasal dari dataset survey dan belum membuktikan bahwa performa yang sama akan tercapai pada data transaksi produksi. Validasi eksternal dan monitoring model tetap diperlukan sebelum deployment produksi.

---

## 📌 Final Takeaway

Project ini menunjukkan penerapan alur **end-to-end Machine Learning** untuk kebutuhan customer retention, mulai dari:

**Data Understanding → EDA → Feature Selection → Preprocessing → Cross-Validation → OOF Threshold Tuning → Holdout Testing → Streamlit Deployment**

Model final yang digunakan adalah **Random Forest** dengan **threshold 0.56**.

Pada holdout test set, model menghasilkan:

- **Churn Recall: 76.5%**
- **Churn F1-Score: 68.4%**
- **ROC-AUC: 0.834**

Model sebaiknya diposisikan sebagai alat **prioritisasi retensi dan decision support**, bukan sebagai sistem yang menyatakan bahwa seorang customer pasti akan Churn.
