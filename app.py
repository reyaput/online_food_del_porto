"""Streamlit app compatible with controlled-threshold train_model.py."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
st.set_page_config(page_title="OrderKu — Prediksi Loyal vs Churn", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    with open(MODELS_DIR / "model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, feature_names, metadata

try:
    model, feature_names, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Artefak model belum ditemukan. Jalankan `py train_model.py` terlebih dahulu.")
    st.stop()
except Exception as exc:
    st.error(f"Gagal memuat artefak model: {exc}")
    st.stop()

CATEGORY_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Marital Status": ["Married", "Single", "Prefer not to say"],
    "Occupation": ["Employee", "House wife", "Self Employeed", "Student"],
    "Monthly Income": ["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"],
    "Educational Qualifications": ["School", "Graduate", "Post Graduate", "Ph.D", "Uneducated"],
    "Feedback": ["Positive", "Negative"],
}
MODEL_INPUT_COLUMNS = [
    "Age", "Gender", "Marital Status", "Occupation",
    "Monthly Income", "Educational Qualifications", "Feedback"
]

def prepare_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Kolom input kurang: {missing}")
    df = df_raw[MODEL_INPUT_COLUMNS].copy()
    for col in CATEGORY_OPTIONS:
        df[col] = df[col].astype(str).str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if df["Age"].isna().any():
        raise ValueError("Kolom Age mengandung nilai yang bukan angka.")
    return df

def predict_customer(df_raw: pd.DataFrame):
    # IMPORTANT: best_model.pkl sekarang FULL sklearn Pipeline.
    # Jangan melakukan pd.get_dummies() di app.
    X = prepare_input(df_raw)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Model classes tidak sesuai: {classes}")
    churn_probability = proba[:, classes.index(0)]
    loyalty_probability = proba[:, classes.index(1)]
    threshold = float(metadata.get("best_churn_threshold", 0.50))
    churn_flag = (churn_probability >= threshold).astype(int)
    return (
        np.round(loyalty_probability * 100, 1),
        np.round(churn_probability * 100, 1),
        churn_flag,
    )

def score_label(score: float):
    if score >= 70:
        return "Loyal Kuat", "green"
    if score >= 40:
        return "Cukup Aman, Perlu Dipantau", "orange"
    return "Berpotensi Churn — Perlu Aksi Segera", "red"

best_name = metadata.get("best_model", "Unknown")
best_test = metadata.get("test_results", {}).get(best_name, {})
best_cv = metadata.get("cv_results", {}).get(best_name, {})
threshold = float(metadata.get("best_churn_threshold", best_test.get("threshold_churn", 0.50)))

def metric(name, fallback=None):
    return best_test.get(name, fallback if fallback is not None else 0)

test_recall_churn = metric("churn_recall", metric("recall"))
test_precision_churn = metric("churn_precision", metric("precision"))
test_f1_churn = metric("churn_f1", metric("f1"))
test_roc_auc = metric("roc_auc")
cv_recall_churn = best_cv.get("churn_recall", {}).get("mean", 0)
cv_f1_churn = best_cv.get("churn_f1", {}).get("mean", 0)
cv_roc_auc = best_cv.get("roc_auc", {}).get("mean", 0)

st.title("🍔 OrderKu — Prediksi Loyal vs Churn")
st.caption(
    f"Model: **{best_name}** · Recall Churn (test): **{test_recall_churn:.0%}** · "
    f"Precision Churn (test): **{test_precision_churn:.0%}** · "
    f"Threshold Churn: **{threshold:.0%}** · Dilatih: {metadata.get('created_at', '')[:10]}"
)
min_precision = metadata.get("business_threshold_rule", {}).get("minimum_churn_precision", 0.40)
st.info(
    f"Threshold Churn dituning dari OOF training predictions dengan batas minimum "
    f"Precision Churn **{min_precision:.0%}**. Test set tetap holdout."
)

tab_manual, tab_batch, tab_explain = st.tabs(["🧍 Input Manual", "📂 Upload CSV (Batch)", "📊 Penjelasan Model"])

with tab_manual:
    st.subheader("Prediksi Satu Customer")
    st.write("Isi profil customer untuk mendapatkan skor loyalitas, probabilitas churn, dan status retensi.")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia", min_value=15, max_value=70, value=25)
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
        marital = st.selectbox("Status Pernikahan", CATEGORY_OPTIONS["Marital Status"])
        occupation = st.selectbox("Pekerjaan", CATEGORY_OPTIONS["Occupation"])
    with col2:
        income = st.selectbox("Pendapatan Bulanan", CATEGORY_OPTIONS["Monthly Income"])
        education = st.selectbox("Pendidikan", CATEGORY_OPTIONS["Educational Qualifications"])
        feedback = st.selectbox("Feedback Terakhir", CATEGORY_OPTIONS["Feedback"])

    if st.button("Hitung Prediksi", type="primary"):
        input_df = pd.DataFrame([{
            "Age": age, "Gender": gender, "Marital Status": marital,
            "Occupation": occupation, "Monthly Income": income,
            "Educational Qualifications": education, "Feedback": feedback,
        }])
        try:
            loyalty, churn, flag = predict_customer(input_df)
            score, churn_prob, is_churn = float(loyalty[0]), float(churn[0]), bool(flag[0])
            label, color = score_label(score)
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Skor Loyalitas", f"{score:.1f}%")
            c2.metric("Probabilitas Churn", f"{churn_prob:.1f}%")
            c3.metric("Threshold Churn", f"{threshold:.0%}")
            st.markdown(f"### :{color}[{label}]")
            if is_churn:
                st.error(f"⚠️ Prioritas Retensi — probabilitas churn {churn_prob:.1f}% >= threshold {threshold:.0%}.")
                st.write("Rekomendasi: cek feedback/keluhan dan pertimbangkan retention offer yang terarah.")
            else:
                st.success("✅ Customer tidak masuk kelompok prioritas churn berdasarkan threshold model.")
                st.write("Rekomendasi: pertahankan engagement; insentif besar tidak menjadi prioritas.")
        except Exception as exc:
            st.error(f"Gagal membuat prediksi: {exc}")

with tab_batch:
    st.subheader("Prediksi Banyak Customer Sekaligus")
    st.write("Upload CSV dengan kolom berikut:")
    st.code(", ".join(MODEL_INPUT_COLUMNS))
    template = pd.DataFrame(columns=MODEL_INPUT_COLUMNS)
    st.download_button("Download Template CSV", template.to_csv(index=False), "template_customer.csv", "text/csv")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            missing = [c for c in MODEL_INPUT_COLUMNS if c not in df_upload.columns]
            if missing:
                st.error(f"Kolom berikut tidak ditemukan: {missing}")
            elif len(df_upload) == 0:
                st.warning("CSV kosong.")
            else:
                with st.spinner("Menghitung skor dan probabilitas churn..."):
                    scores, churn_probs, flags = predict_customer(df_upload)
                result = df_upload.copy()
                result["Skor_Loyalitas_%"] = scores
                result["Probabilitas_Churn_%"] = churn_probs
                result["Status_Churn"] = np.where(flags == 1, "Berpotensi Churn", "Tidak Prioritas Churn")
                result["Kategori_Loyalitas"] = [score_label(float(s))[0] for s in scores]
                st.success(f"Berhasil memproses {len(result)} customer.")
                total = len(result); churn_count = int(flags.sum())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customer", total)
                c2.metric("Prioritas Churn", churn_count)
                c3.metric("Bukan Prioritas", total - churn_count)
                c4.metric("Rata-rata Loyalitas", f"{np.mean(scores):.1f}%")
                st.dataframe(result.sort_values("Probabilitas_Churn_%", ascending=False), use_container_width=True)
                st.download_button("Download Hasil (CSV)", result.to_csv(index=False), "hasil_prediksi_loyalitas_churn.csv", "text/csv", type="primary")
        except Exception as exc:
            st.error(f"Gagal memproses CSV: {exc}")

with tab_explain:
    st.subheader("Penjelasan Model")
    st.markdown("#### Ringkasan Performa")
    cols = st.columns(5)
    cols[0].metric("CV Recall Churn", f"{cv_recall_churn:.0%}")
    cols[1].metric("CV F1 Churn", f"{cv_f1_churn:.0%}")
    cols[2].metric("CV ROC-AUC", f"{cv_roc_auc:.3f}")
    cols[3].metric("Test Recall Churn", f"{test_recall_churn:.0%}")
    cols[4].metric("Test Precision Churn", f"{test_precision_churn:.0%}")

    st.markdown("---")
    st.markdown("#### Threshold Keputusan")
    st.write(f"Threshold Churn: **{threshold:.0%}**")
    st.write(f"Customer diprioritaskan untuk retensi ketika probabilitas churn >= **{threshold:.0%}**.")
    st.caption(f"Constraint threshold: Precision Churn minimum {min_precision:.0%}.")

    st.markdown("#### Faktor Paling Berpengaruh Secara Umum")
    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        names = preprocessor.get_feature_names_out()
        if hasattr(estimator, "feature_importances_"):
            fi = pd.DataFrame({"Fitur": names, "Kepentingan": estimator.feature_importances_})
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_[0]
            fi = pd.DataFrame({"Fitur": names, "Koefisien": coef, "Kepentingan": np.abs(coef)})
        else:
            fi = None
        if fi is not None:
            fi = fi.sort_values("Kepentingan", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(fi["Fitur"][::-1], fi["Kepentingan"][::-1])
            ax.set_xlabel("Kepentingan")
            ax.set_title("Top 10 Faktor Paling Berpengaruh")
            plt.tight_layout()
            st.pyplot(fig)
            if "Koefisien" in fi.columns:
                st.caption("Pada Logistic Regression, nilai absolut koefisien dipakai sebagai ukuran kepentingan; tandanya menunjukkan arah terhadap Loyal (Target=1).")
    except Exception as exc:
        st.warning(f"Feature importance tidak dapat ditampilkan: {exc}")

    st.markdown("---")
    st.markdown("#### Keterbatasan Model")
    st.markdown(
        f"""- Model dilatih dari data survey, bukan transaksi riil; hasil menunjukkan pola prediksi, bukan bukti sebab-akibat.\n"
        f"- Test Precision Churn sekitar **{test_precision_churn:.0%}** dan Recall Churn sekitar **{test_recall_churn:.0%}**; keduanya perlu dibaca bersama.\n"
        "- Model membutuhkan Feedback sehingga tidak cocok untuk customer tanpa feedback.\n"
        "- Family size dan Customer Type sengaja tidak dipakai karena alasan leakage/definisi target.\n"
        "- Pin code, latitude, dan longitude tidak digunakan berdasarkan analisis sebelumnya."""
    )
    with st.expander("Lihat detail metadata model"):
        st.json(metadata)