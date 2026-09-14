**# 🍔 Online Food Delivery Customer Churn Prediction System (OrderKu)**

[![Python]\(https\://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)]\(https\://www\.python.org/)

[![Streamlit]\(https\://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]\(https\://streamlit.io/)

[![Scikit-Learn]\(https\://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)]\(https\://scikit-learn.org/)

[![Pandas]\(https\://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)]\(https\://pandas.pydata.org/)

**---**

**## 📋 Table of Contents**

\- [Project Overview]\(#-project-overview)

\- [Business Understanding]\(#-business-understanding)

\- [Installation & Setup]\(#-installation--setup)

\- [How to Run]\(#-how-to-run)

\- [Data Understanding]\(#-data-understanding)

\- [Key Insights (EDA)]\(#-key-insights-eda)

\- [Machine Learning Model]\(#-machine-learning-model)

\- [Business Recommendations]\(#-business-recommendations)

**---**

**## 📋 Project Overview**

Proyek ini bertujuan untuk membangun portofolio **\*\*Data Analysis & End-to-End Machine Learning\*\*** dalam industri *\*Online Food Delivery\** (**\*\*OrderKu\*\***). Sistem ini dirancang untuk memprediksi pelanggan yang berisiko berhenti memesan (*\*Churn\** / \`Output = No\`) berdasarkan profil demografis, latar belakang ekonomi, dan umpan balik (*\*Feedback\**) yang diberikan.

Dengan memadukan eksplorasi data mendalam (*\*EDA\**), pencegahan kebocoran data (*\*data leakage prevention\**), *\*threshold tuning\** dengan batasan bisnis (*\*business constraint\**), serta antarmuka interaktif berbasis **\*\*Streamlit\*\***, tim operasional dan CRM dapat melakukan intervensi retensi secara cepat, terukur, dan efisien sebelum pelanggan beralih ke platform kompetitor.

**---**

**## 💼 Business Understanding**

**### Problem Statement**

Dalam industri pesan-antar makanan daring (*\*food delivery\**), kompetisi sangat ketat dengan biaya akuisisi pelanggan (*\*Customer Acquisition Cost\** / CAC) yang tinggi akibat perang promo dan diskon. 

\* Dataset menunjukkan tingkat *\*Churn\** sebesar **\*\*\~22.4%\*\*** (hampir 1 dari 4 pelanggan berhenti memesan).

\* Mengakuisisi pelanggan baru membutuhkan biaya **\*\*5x hingga 7x lipat\*\*** dibandingkan mempertahankan pelanggan lama.

\* Pelanggan yang berhenti memesan menurunkan nilai transaksi bruto (*\*Gross Merchandise Value\** / GMV) serta merugikan ekosistem merchant dan mitra pengemudi.

**### Objectives**

1\. **\*\*Identifikasi Faktor Risiko (Root Cause Analysis):\*\*** Mengungkap pola demografis, sosio-ekonomi, dan sinyal kepuasan yang mendorong pelanggan berhenti menggunakan layanan.

2\. **\*\*Pencegahan Data Leakage & Pemodelan Robust:\*\*** Merancang pipeline Machine Learning yang bebas dari kebocoran fitur dengan fokus evaluasi pada deteksi kelas minoritas (*\*Churn Recall & Churn Precision\**).

3\. **\*\*Threshold Tuning dengan Batasan Bisnis:\*\*** Mengoptimalkan *\*decision threshold\** dari probabilitas prediksi dengan syarat *\*Precision Churn\** $\ge$ 40%, sehingga tim retensi tidak membuang anggaran promo ke pelanggan yang sebenarnya loyal.

4\. **\*\*Dashboard Operasional Interaktif:\*\*** Menyediakan aplikasi web (Streamlit) yang mendukung pengecekan risiko pelanggan satuan (*\*single prediction\**) maupun pemrosesan massal (*\*batch CSV upload\**) untuk kebutuhan tim CRM.

**---**

**## 🛠️ Installation & Setup**

Proyek ini dibangun menggunakan bahasa pemrograman Python. Disarankan menggunakan **\*\*Conda\*\*** atau **\*\*Virtualenv\*\*** untuk manajemen environment agar dependensi terisolasi dengan rapi.

**### 1. Prerequisite**

Pastikan Anda telah menginstal [Anaconda]\(https\://www\.anaconda.com/) / [Miniconda]\(https\://docs.conda.io/en/latest/miniconda.html) atau Python 3.13.

**### 2. Setup Environment**

Buka terminal (Anaconda Prompt / CMD / PowerShell) dan jalankan perintah berikut:

\`\`\`bash

\# 1. Buat environment baru bernama 'food\_churn' dengan Python 3.11

conda create -n food\_churn python=3.11 -y

\# 2. Aktifkan environment

conda activate food\_churn

\# 3. Masuk ke direktori project (sesuaikan path folder Anda)

cd "path/to/online\_food\_del\_porto"

\`\`\`

**### 3. Install Dependencies**

Install library yang dibutuhkan menggunakan pip:

\`\`\`bash

pip install -r requirements.txt

\`\`\`

\> File \`requirements.txt\` mencakup dependensi utama: \`streamlit\`, \`pandas\`, \`numpy\`, \`scikit-learn\`, \`matplotlib\`, \`seaborn\`, dan \`joblib\`.

**---**

**## 🚀 How to Run**

Proyek ini menyediakan alur kerja yang mudah dijalankan, mulai dari pelatihan ulang model (*\*training pipeline\**) hingga deployment aplikasi web (*\*Streamlit dashboard\**).

**### A. Melatih Ulang Model (Training Pipeline)**

Jalankan skrip ini jika ingin melatih ulang model, memvalidasi cross-validation, mencari threshold optimal, atau memperbarui artefak metadata:

\`\`\`bash

python train\_model.py

\`\`\`

\* **\*\*Proses yang dijalankan:\*\***

  1. Membaca dataset mentah (\`data/raw/online\_food\_delivery\_dataset.csv\`).

  2. Melakukan data cleaning & menyimpan data perantara (\`data/interim/df\_interim.csv\` & \`data/processed/df\_model\_ready.csv\`).

  3. Melakukan 5-Fold Stratified Cross-Validation pada Logistic Regression dan Random Forest.

  4. Melakukan OOF (*\*Out-Of-Fold\**) threshold tuning dengan constraint bisnis (*\*Precision Churn\** $\ge$ 40%).

  5. Menguji performa final pada Holdout Test Set (20%).

  6. Menyimpan model pipeline (\`best\_model.pkl\`), feature names (\`feature\_names.pkl\`), dan rekam jejak eksperimen lengkap (\`model\_metadata.json\`) ke folder \`models/\`.

**### B. Menjalankan Dashboard Prediksi (Streamlit App)**

Jalankan perintah ini untuk membuka antarmuka aplikasi prediksi berbasis web:

\`\`\`bash

streamlit run app.py

\`\`\`

\* **\*\*Akses Aplikasi:\*\*** Browser akan terbuka secara otomatis di \`http\://localhost:8501\`.

\* **\*\*Fitur Utama Aplikasi:\*\***

  \* **\*\*🧍 Input Manual:\*\*** Masukkan parameter profil pelanggan satu per satu untuk mendapatkan skor loyalitas (%), probabilitas churn (%), dan label status retensi secara real-time.

  \* **\*\*📂 Upload CSV (Batch):\*\*** Unggah file CSV pelanggan secara massal, lengkapi dengan template unduhan, dan ekspor hasil prediksi beserta rekomendasi prioritas retensi.

  \* **\*\*📊 Penjelasan Model:\*\*** Visualisasi transparansi model mencakup metrik evaluasi CV & Test, threshold keputusan, serta grafik *\*Top 10 Feature Importance\**.

**---**

**## 📊 Data Understanding**

Dataset yang digunakan mencakup data survei pelanggan layanan pesan-antar makanan daring dengan **\*\*388 baris\*\*** dan **\*\*14 fitur\*\***.

**### Kamus Data (Feature Dictionary)**

\| Fitur | Deskripsi | Tipe Data | Status dalam Model |

\| :--- | :--- | :--- | :--- |

\| **\*\*Age\*\*** | Usia pelanggan (rentang 18 - 33 tahun) | Numerik | Digunakan (Input) |

\| **\*\*Gender\*\*** | Jenis kelamin (\`Male\`, \`Female\`) | Kategorikal | Digunakan (Input) |

\| **\*\*Marital Status\*\*** | Status pernikahan (\`Single\`, \`Married\`, \`Prefer not to say\`) | Kategorikal | Digunakan (Input) |

\| **\*\*Occupation\*\*** | Status pekerjaan (\`Student\`, \`Employee\`, \`Self Employeed\`, \`House wife\`) | Kategorikal | Digunakan (Input) |

\| **\*\*Monthly Income\*\*** | Rentang pendapatan bulanan (\`No Income\`, \`Below Rs.10000\`, \`10001 to 25000\`, \`25001 to 50000\`, \`More than 50000\`) | Kategorikal | Digunakan (Input) |

\| **\*\*Educational Qualifications\*\*** | Tingkat pendidikan formal (\`School\`, \`Graduate\`, \`Post Graduate\`, \`Ph.D\`, \`Uneducated\`) | Kategorikal | Digunakan (Input) |

\| **\*\*Feedback\*\*** | Sentimen ulasan terakhir yang diberikan pelanggan (\`Positive\`, \`Negative\`) | Kategorikal | Digunakan (Input) |

\| **\*\*Output\*\*** | Status pemesanan kembali (\`Yes\` = Loyal/Stay, \`No\` = Churn) | Target | **\*\*Target Prediksi\*\*** |

\| **\*\*Family size\*\*** | Jumlah anggota keluarga | Numerik | *\*Excluded\** (Penyebab data leakage terhadap customer profile) |

\| **\*\*Customer Type\*\*** | Tipe pelanggan | Kategorikal | *\*Excluded\** (Beririsan langsung dengan definisi label) |

\| **\*\*latitude\*\*** | Koordinat lintang tempat tinggal | Numerik | *\*Excluded\** (Tidak relevan untuk profil retensi umum) |

\| **\*\*longitude\*\*** | Koordinat bujur tempat tinggal | Numerik | *\*Excluded\** (Tidak relevan untuk profil retensi umum) |

\| **\*\*Pin code\*\*** | Kode pos area pengiriman | Numerik | *\*Excluded\** (Kardinalitas lokasi lokal) |

**---**

**## 📊 Key Insights (EDA)**

Berdasarkan analisis eksplorasi data terhadap 388 pelanggan, berikut adalah temuan utama:

**### 1. Tingkat Retensi & Churn Keseluruhan**

\* **\*\*Distribusi Target:\*\*** Sebanyak **\*\*77.6% (301 pelanggan)\*\*** berstatus Loyal (\`Output = Yes\`), sedangkan **\*\*22.4% (87 pelanggan)\*\*** berstatus Churn (\`Output = No\`).

\* Meskipun kelas mayoritas adalah pelanggan setia, tingkat churn sebesar 22.4% menunjukkan bahwa lebih dari 1 dari 5 pelanggan berhenti menggunakan layanan.

**### 2. Sinyal Ulasan (Feedback) Adalah Indikator Terkuat**

\* Pelanggan yang memberikan **\*\*Feedback "Negative"\*\*** memiliki tingkat Churn fantastis sebesar **\*\*74.6%\*\***.

\* Sebaliknya, pelanggan dengan **\*\*Feedback "Positive"\*\*** mencatat tingkat retensi/loyalitas sebesar **\*\*89.3%\*\*** (tingkat churn hanya 10.7%).

\* *\*Insight:\** Ulasan negatif bukan sekadar komplain, melainkan sinyal langsung bahwa pelanggan berada di ambang churn.

**### 3. Dinamika Pekerjaan & Penghasilan Bulanan**

\* **\*\*Mahasiswa & Tanpa Penghasilan Sangat Loyal:\*\*** Kelompok \`Student\` (tingkat loyalitas **\*\*88.9%\*\***) dan \`No Income\` (loyalitas **\*\*87.7%\*\***) merupakan segmen pengguna paling stabil, umumnya mengandalkan layanan untuk kepraktisan makan harian.

\* **\*\*Pekerja & Profesional Lebih Rentan Churn:\*\*** Kelompok \`Employee\` (churn **\*\*35.6%\*\***) dan \`Self Employeed\` (churn **\*\*37.0%\*\***), terutama pada rentang penghasilan \`25001 to 50000\` (churn **\*\*39.1%\*\***), memiliki tingkat churn tertinggi.

\* *\*Insight:\** Kelompok berpenghasilan memiliki ekspektasi layanan lebih tinggi (kecepatan antar, kualitas kemasan) serta sensitif terhadap pengalaman yang mengecewakan.

**### 4. Pengaruh Status Pernikahan & Usia**

\* **\*\*Status Pernikahan:\*\*** Pelanggan yang berstatus \`Married\` memiliki churn rate sebesar **\*\*38.9%\*\***, jauh lebih tinggi dibandingkan pelanggan \`Single\` yang hanya mencatat churn rate **\*\*14.6%\*\***.

\* **\*\*Usia:\*\*** Pelanggan yang Churn memiliki rata-rata usia yang sedikit lebih tua (**\*\*26.0 tahun\*\***) dibandingkan pelanggan Loyal (**\*\*24.2 tahun\*\***). Pelanggan berkeluarga cenderung beralih memasak di rumah jika layanan pesan-antar tidak memenuhi standar higienis atau ketepatan waktu.

**---**

**## 🤖 Machine Learning Model

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

Dua algoritma dibandingkan menggunakan **5-Fold Stratified Cross-Validation**:

- Logistic Regression
- Random Forest Classifier

Hasil CV terbaru:

| Model | CV Recall Churn | CV Precision Churn | CV F1 Churn | CV ROC-AUC |
|---|---:|---:|---:|---:|
| **Random Forest** | **70.0% ± 8.3%** | 56.6% ± 21.8% | **59.5% ± 6.8%** | **0.839 ± 0.057** |
| Logistic Regression | 64.3% ± 6.4% | **57.3% ± 16.0%** | 59.1% ± 6.1% | 0.824 ± 0.053 |

**Random Forest dipilih sebagai model final** karena memiliki CV Recall Churn, CV F1 Churn, dan CV ROC-AUC yang sedikit lebih tinggi.

### 3. OOF Threshold Tuning dengan Business Constraint

Selain threshold default 0.50, project ini melakukan **Out-of-Fold (OOF) threshold tuning** pada training set.

Constraint bisnis:

```text
Precision Churn >= 40%
```

Tujuannya adalah mencegah threshold terlalu rendah sehingga terlalu banyak customer loyal salah ditandai sebagai churn.

Hasil tuning OOF:

| Model | Threshold | Recall Churn | Precision Churn | F1 Churn | Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.63 | 61.4% | **67.2%** | **64.2%** | **84.5%** |
| **Random Forest** | **0.56** | **65.7%** | 61.3% | 63.4% | 82.9% |

Threshold final untuk Random Forest adalah:

```text
0.56 (56%)
```

Threshold dipilih menggunakan training-set OOF prediction. **Holdout test set tidak digunakan untuk memilih threshold.**

### 4. Final Holdout Test Set

Data dibagi menjadi:

```text
Train = 310 customer
Test  = 78 customer
```

Test set tetap disimpan sebagai data yang tidak digunakan dalam pemilihan model maupun threshold.

Dengan **Random Forest + threshold Churn 0.56**, hasil test:

| Metrik Evaluasi | Nilai | Interpretasi |
|---|---:|---|
| **Accuracy** | **84.6%** | Proporsi prediksi benar pada keseluruhan test set. |
| **Churn Recall** | **76.5%** | Berhasil mendeteksi **13 dari 17** customer yang sebenarnya churn. |
| **Churn Precision** | **61.9%** | Sekitar 62% customer yang ditandai churn benar-benar churn. |
| **Churn F1-Score** | **68.4%** | Keseimbangan precision dan recall untuk kelas Churn. |
| **Recall Loyal** | **86.9%** | Sebagian besar customer Loyal tetap dikenali sebagai Loyal. |
| **Loyal F1-Score** | **89.8%** | Performa klasifikasi kelas Loyal. |
| **ROC-AUC** | **0.834** | Kemampuan model membedakan Loyal dan Churn berdasarkan ranking probabilitas. |
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

Hasil model menunjukkan trade-off yang cukup seimbang.

**Recall Churn 76.5%** berarti sebagian besar customer yang benar-benar churn berhasil masuk ke daftar prioritas retensi.

**Precision Churn 61.9%** berarti model tidak sekadar menandai hampir semua customer sebagai churn; mayoritas customer yang ditandai memang berasal dari kelas Churn.

Dengan demikian, model lebih cocok digunakan sebagai **decision-support / prioritization tool** untuk tim CRM, bukan sebagai keputusan otomatis bahwa seorang customer pasti churn.

## 💡 Business Recommendations

### 1. Gunakan Model sebagai Prioritas Retensi

Gunakan probabilitas churn untuk membuat daftar prioritas pelanggan yang perlu ditinjau oleh tim CRM.

Dengan threshold **56%**, customer dengan probabilitas churn minimal 56% masuk kelompok prioritas retensi.

### 2. Hindari Pendekatan “Spray and Pray”

Model membantu membatasi intervensi agar tidak diberikan secara merata kepada seluruh customer.

Tujuannya adalah mengarahkan anggaran retensi kepada customer yang memiliki risiko churn lebih tinggi.

### 3. Gunakan Feedback sebagai Sinyal Operasional

`Feedback` merupakan salah satu fitur model. Feedback dapat digunakan sebagai konteks tambahan ketika tim CRM menentukan jenis tindak lanjut yang sesuai.

Contohnya, customer dengan pengalaman negatif dapat diarahkan ke proses recovery atau penanganan keluhan.

### 4. Validasi dengan Data Produksi

Sebelum digunakan untuk otomatisasi kampanye, model sebaiknya divalidasi menggunakan data transaksi nyata seperti:

- frekuensi order,
- recency transaksi,
- monetary value,
- jumlah order sebelumnya,
- histori penggunaan promo,
- histori komplain.

Data transaksi tersebut dapat membantu meningkatkan kemampuan model dalam menggambarkan perilaku customer secara aktual.


## ⚠️ Limitations

1. **Dataset kecil**  
   Dataset terdiri dari 388 observasi dan test set 78 observasi. Metric pada satu split dapat berubah apabila pembagian data berubah.

2. **Data survey**  
   Dataset berbasis survey, bukan data transaksi produksi. Model menunjukkan kemampuan prediksi pada dataset ini dan belum membuktikan hubungan sebab-akibat.

3. **Feedback diperlukan**  
   Model memakai `Feedback` sebagai salah satu fitur sehingga penggunaannya terbatas pada customer yang memiliki feedback.

4. **Threshold bukan kepastian churn**  
   Threshold 56% adalah batas keputusan model untuk prioritas retensi, bukan berarti customer pasti akan churn.

5. **Duplicate rows**  
   Sebanyak 103 duplicate rows terdeteksi dan tidak di-drop sesuai keputusan analisis dataset. Pada data produksi dengan customer ID unik, pemeriksaan duplicate sebaiknya dilakukan kembali.

