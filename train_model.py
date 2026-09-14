

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "online_food_delivery_dataset.csv"
INTERIM_DATA_PATH = BASE_DIR / "data" / "interim" / "df_interim.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "df_model_ready.csv"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Business constraint:
# model tidak boleh memilih threshold dengan Precision Churn < 40%.
MIN_CHURN_PRECISION = 0.40

# Threshold search.
THRESHOLD_GRID = np.round(np.arange(0.20, 0.81, 0.01), 2)


class LoyaltyChurnPredictor:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}
        self.cv_results = {}
        self.oof_results = {}
        self.test_results = {}

        self.best_model_name = None
        self.best_model = None
        self.best_churn_threshold = 0.50

        self.categorical_features = [
            "Gender",
            "Marital Status",
            "Occupation",
            "Monthly Income",
            "Educational Qualifications",
            "Feedback",
        ]
        self.numerical_features = ["Age"]

        self.feature_names = None

    # ========================================================
    # 1. LOAD
    # ========================================================
    def load_data(self, path: Path = RAW_DATA_PATH):
        print(f"[1/8] Loading data dari {path} ...")

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan:\n{path}"
            )

        df = pd.read_csv(path)
        df = df.drop(columns="Unnamed: 13", errors="ignore")

        self.df = df

        print(f"       Shape awal: {df.shape}")
        print(f"       Kolom: {len(df.columns)}")

        return self

    # ========================================================
    # 2. CLEAN
    # ========================================================
    def clean_data(self):
        print("[2/8] Cleaning data ...")

        df = self.df.copy()

        required_columns = (
            self.categorical_features
            + self.numerical_features
            + ["Output"]
        )

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {missing}"
            )

        # Standardize strings.
        for col in self.categorical_features:
            df[col] = df[col].astype(str).str.strip()

        df["Output"] = df["Output"].astype(str).str.strip()
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

        # Remove rows that cannot be modeled.
        before = len(df)
        df = df.dropna(subset=["Age", "Output"])
        dropped_missing = before - len(df)

        valid_output = {"Yes", "No"}
        invalid_output = sorted(set(df["Output"]) - valid_output)

        if invalid_output:
            raise ValueError(
                f"Nilai Output tidak valid: {invalid_output}. "
                "Harus berupa Yes/No."
            )

        n_dupes = int(df.duplicated().sum())

        print(
            f"       Baris duplicate terdeteksi: {n_dupes} "
            f"(tidak di-drop)"
        )
        print(
            f"       Baris dibuang karena missing wajib: "
            f"{dropped_missing}"
        )

        INTERIM_DATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        df.to_csv(
            INTERIM_DATA_PATH,
            index=False,
        )

        self.df = df

        return self

    # ========================================================
    # 3. TARGET
    # ========================================================
    def build_target(self):
        print("[3/8] Membuat target variable ...")

        # Asli:
        # 1 = Loyal
        # 0 = Churn
        self.df["Target"] = (
            self.df["Output"] == "Yes"
        ).astype(int)

        dist = self.df["Target"].value_counts(
            normalize=True
        ) * 100

        print(
            f"       Distribusi target -> "
            f"Loyal(1): {dist.get(1, 0):.1f}% | "
            f"Churn(0): {dist.get(0, 0):.1f}%"
        )

        print(
            "       Evaluasi bisnis: Churn (kelas 0) "
            "diposisikan sebagai positive class."
        )

        return self

    # ========================================================
    # 4. FEATURES
    # ========================================================
    def build_features(self):
        print("[4/8] Menyiapkan fitur ...")

        cols = (
            self.categorical_features
            + self.numerical_features
        )

        self.X = self.df[cols].copy()
        self.y = self.df["Target"].copy()

        PROCESSED_DATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Simpan raw model-ready data.
        # Encoding dilakukan di Pipeline setelah split.
        processed = self.X.copy()
        processed["Target"] = self.y.values

        processed.to_csv(
            PROCESSED_DATA_PATH,
            index=False,
        )

        print(f"       Fitur mentah: {len(cols)}")
        print(
            f"       Categorical: "
            f"{len(self.categorical_features)}"
        )
        print(
            f"       Numerical: "
            f"{len(self.numerical_features)}"
        )

        return self

    # ========================================================
    # 5. SPLIT
    # ========================================================
    def split_data(self):
        print(
            f"[5/8] Train-test split "
            f"(test_size={TEST_SIZE}, stratify=y) ..."
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=self.y,
        )

        print(
            f"       Train: {len(self.X_train)} baris | "
            f"Test: {len(self.X_test)} baris"
        )

        train_counts = (
            self.y_train.value_counts().to_dict()
        )

        print(
            f"       Train target -> "
            f"Churn(0): {train_counts.get(0, 0)} | "
            f"Loyal(1): {train_counts.get(1, 0)}"
        )

        return self

    # ========================================================
    # PREPROCESSOR
    # ========================================================
    def _build_preprocessor(self):
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
            )
        except TypeError:
            # Compatibility sklearn lama.
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse=False,
            )

        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    encoder,
                    self.categorical_features,
                ),
                (
                    "num",
                    "passthrough",
                    self.numerical_features,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # ========================================================
    # MODELS
    # ========================================================
    def _build_models(self):
        lr = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self._build_preprocessor(),
                ),
                (
                    "model",
                    LogisticRegression(
                        random_state=self.random_state,
                        max_iter=2000,
                        class_weight="balanced",
                        C=0.5,
                    ),
                ),
            ]
        )

        rf = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self._build_preprocessor(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_split=12,
                        min_samples_leaf=6,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        return {
            "Logistic Regression": lr,
            "Random Forest": rf,
        }

    # ========================================================
    # 6. TRAIN + CV
    # ========================================================
    def train_models(self):
        print(
            "[6/8] Training & membandingkan model "
            "(Logistic Regression, Random Forest) ..."
        )

        self.models = self._build_models()

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=self.random_state,
        )

        # IMPORTANT:
        # sklearn default recall = positive class 1.
        # Kita explicitly set pos_label=0 -> Churn.
        churn_recall_scorer = make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0,
        )

        churn_precision_scorer = make_scorer(
            precision_score,
            pos_label=0,
            zero_division=0,
        )

        churn_f1_scorer = make_scorer(
            f1_score,
            pos_label=0,
            zero_division=0,
        )

        scoring = {
            "churn_recall": churn_recall_scorer,
            "churn_precision": churn_precision_scorer,
            "churn_f1": churn_f1_scorer,
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
        }

        for name, model in self.models.items():
            scores = cross_validate(
                model,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
            )

            self.cv_results[name] = {
                metric: {
                    "mean": float(
                        np.mean(
                            scores[f"test_{metric}"]
                        )
                    ),
                    "std": float(
                        np.std(
                            scores[f"test_{metric}"]
                        )
                    ),
                }
                for metric in scoring
            }

            print(f"\n       {name}:")
            print(
                "         CV Recall Churn = "
                f"{self.cv_results[name]['churn_recall']['mean']:.3f} "
                f"(+/-{self.cv_results[name]['churn_recall']['std']:.3f})"
            )
            print(
                "         CV Precision Churn = "
                f"{self.cv_results[name]['churn_precision']['mean']:.3f} "
                f"(+/-{self.cv_results[name]['churn_precision']['std']:.3f})"
            )
            print(
                "         CV F1 Churn = "
                f"{self.cv_results[name]['churn_f1']['mean']:.3f} "
                f"(+/-{self.cv_results[name]['churn_f1']['std']:.3f})"
            )
            print(
                "         CV ROC-AUC = "
                f"{self.cv_results[name]['roc_auc']['mean']:.3f} "
                f"(+/-{self.cv_results[name]['roc_auc']['std']:.3f})"
            )

        # ----------------------------------------------------
        # OOF threshold tuning
        # ----------------------------------------------------
        print(
            "\n       Threshold tuning dari OOF training "
            f"dengan Precision Churn >= "
            f"{MIN_CHURN_PRECISION:.2f}"
        )

        for name, model in self.models.items():
            oof_proba_loyal = cross_val_predict(
                model,
                self.X_train,
                self.y_train,
                cv=cv,
                method="predict_proba",
            )[:, 1]

            # Probability of Churn = 1 - P(Loyal)
            oof_proba_churn = 1.0 - oof_proba_loyal

            result = self._select_controlled_threshold(
                self.y_train,
                oof_proba_churn,
            )

            # PR curve untuk metadata/analisis.
            churn_actual = (
                self.y_train == 0
            ).astype(int)

            pr_precision, pr_recall, pr_thresholds = (
                precision_recall_curve(
                    churn_actual,
                    oof_proba_churn,
                )
            )

            self.oof_results[name] = {
                **result,
                "average_precision": float(
                    average_precision_score(
                        churn_actual,
                        oof_proba_churn,
                    )
                ),
                "pr_curve": {
                    "precision": pr_precision.tolist(),
                    "recall": pr_recall.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                },
            }

            print(
                f"\n       {name} OOF:"
                f"\n         Threshold = "
                f"{result['threshold']:.2f}"
                f"\n         Recall Churn = "
                f"{result['recall']:.3f}"
                f"\n         Precision Churn = "
                f"{result['precision']:.3f}"
                f"\n         F1 Churn = "
                f"{result['f1']:.3f}"
                f"\n         Accuracy = "
                f"{result['accuracy']:.3f}"
                f"\n         Constraint satisfied = "
                f"{result['precision'] >= MIN_CHURN_PRECISION}"
            )

        # Fit final models ke seluruh train set.
        for model in self.models.values():
            model.fit(
                self.X_train,
                self.y_train,
            )

        return self

    # ========================================================
    # CONTROLLED THRESHOLD SELECTION
    # ========================================================
    def _select_controlled_threshold(
        self,
        y_true,
        churn_probability,
    ):
        y_churn = (
            np.asarray(y_true) == 0
        ).astype(int)

        candidates = []

        for threshold in THRESHOLD_GRID:
            pred_churn = (
                churn_probability >= threshold
            ).astype(int)

            precision = precision_score(
                y_churn,
                pred_churn,
                zero_division=0,
            )

            recall = recall_score(
                y_churn,
                pred_churn,
                zero_division=0,
            )

            f1 = f1_score(
                y_churn,
                pred_churn,
                zero_division=0,
            )

            accuracy = accuracy_score(
                y_churn,
                pred_churn,
            )

            # Treat NaN/inf robustly.
            values = [
                precision,
                recall,
                f1,
                accuracy,
            ]

            if not all(np.isfinite(values)):
                continue

            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float(accuracy),
                    "confusion_matrix": (
                        confusion_matrix(
                            y_churn,
                            pred_churn,
                        ).tolist()
                    ),
                }
            )

        constrained = [
            c for c in candidates
            if c["precision"] >= MIN_CHURN_PRECISION
        ]

        pool = (
            constrained
            if constrained
            else candidates
        )

        if not pool:
            raise RuntimeError(
                "Tidak ada kandidat threshold yang dapat dievaluasi."
            )

        best = max(
            pool,
            key=lambda c: (
                c["f1"],
                c["recall"],
                c["accuracy"],
                c["threshold"],
            ),
        )

        best["constraint_used"] = bool(
            constrained
        )

        best["min_precision_required"] = (
            MIN_CHURN_PRECISION
        )

        return best

    # ========================================================
    # 7. EVALUATE
    # ========================================================
    def evaluate_and_select(self):
        print(
            "\n[7/8] Evaluasi test set & memilih model "
            "berdasarkan cross-validation ..."
        )

        for name, model in self.models.items():
            proba_loyal = model.predict_proba(
                self.X_test
            )[:, 1]

            proba_churn = 1.0 - proba_loyal

            threshold = self.oof_results[name][
                "threshold"
            ]

            # 1 = Churn probability above threshold.
            pred_churn = (
                proba_churn >= threshold
            ).astype(int)

            # Back to original Target:
            # pred_churn=1 => Target=0 => Churn
            # pred_churn=0 => Target=1 => Loyal
            y_pred = np.where(
                pred_churn == 1,
                0,
                1,
            )

            churn_precision = precision_score(
                self.y_test,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
            churn_recall = recall_score(
                self.y_test,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
            churn_f1 = f1_score(
                self.y_test,
                y_pred,
                pos_label=0,
                zero_division=0,
            )

            loyal_precision = precision_score(
                self.y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
            loyal_recall = recall_score(
                self.y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
            loyal_f1 = f1_score(
                self.y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            )

            accuracy = accuracy_score(
                self.y_test,
                y_pred,
            )

            roc_auc = roc_auc_score(
                self.y_test,
                proba_loyal,
            )

            average_precision = (
                average_precision_score(
                    (self.y_test == 0).astype(int),
                    proba_churn,
                )
            )

            cm = confusion_matrix(
                self.y_test,
                y_pred,
            ).tolist()

            self.test_results[name] = {
                "threshold_churn": float(threshold),
                "accuracy": float(accuracy),
                "churn_precision": float(churn_precision),
                "churn_recall": float(churn_recall),
                "churn_f1": float(churn_f1),
                "loyal_precision": float(loyal_precision),
                "loyal_recall": float(loyal_recall),
                "loyal_f1": float(loyal_f1),
                "roc_auc": float(roc_auc),
                "average_precision_churn": float(
                    average_precision
                ),
                "confusion_matrix": cm,
            }

            print(
                f"\n--- {name} (test set) ---"
            )
            print(
                f"Threshold Churn : {threshold:.2f}"
            )
            print(
                f"Accuracy        : {accuracy:.3f}"
            )
            print(
                f"Recall Churn    : {churn_recall:.3f}"
            )
            print(
                f"Precision Churn : {churn_precision:.3f}"
            )
            print(
                f"F1 Churn        : {churn_f1:.3f}"
            )
            print(
                f"Recall Loyal    : {loyal_recall:.3f}"
            )
            print(
                f"F1 Loyal        : {loyal_f1:.3f}"
            )
            print(
                f"ROC-AUC         : {roc_auc:.3f}"
            )
            print(
                f"AP Churn        : {average_precision:.3f}"
            )

            print("\nClassification report:")
            print(
                classification_report(
                    self.y_test,
                    y_pred,
                    target_names=[
                        "Churn",
                        "Loyal",
                    ],
                    zero_division=0,
                )
            )

            print(
                f"Confusion matrix: {cm}"
            )

        # ----------------------------------------------------
        # Select model ONLY from CV.
        # Test results are not used as selection criteria.
        # ----------------------------------------------------
        self.best_model_name = max(
            self.cv_results.keys(),
            key=lambda name: (
                self.cv_results[name][
                    "churn_recall"
                ]["mean"],
                self.cv_results[name][
                    "churn_f1"
                ]["mean"],
                self.cv_results[name][
                    "roc_auc"
                ]["mean"],
            ),
        )

        self.best_model = self.models[
            self.best_model_name
        ]

        self.best_churn_threshold = self.oof_results[
            self.best_model_name
        ]["threshold"]

        cv_best = self.cv_results[
            self.best_model_name
        ]
        test_best = self.test_results[
            self.best_model_name
        ]

        print(
            "\n>>> MODEL TERPILIH: "
            f"{self.best_model_name}"
        )
        print(
            "    Dasar pemilihan: "
            "CV Recall Churn"
        )
        print(
            f"    CV Recall Churn = "
            f"{cv_best['churn_recall']['mean']:.3f}"
        )
        print(
            f"    CV F1 Churn = "
            f"{cv_best['churn_f1']['mean']:.3f}"
        )
        print(
            f"    CV ROC-AUC = "
            f"{cv_best['roc_auc']['mean']:.3f}"
        )
        print(
            f"    Threshold Churn = "
            f"{self.best_churn_threshold:.2f}"
        )
        print(
            f"    Test Precision Churn = "
            f"{test_best['churn_precision']:.3f}"
        )
        print(
            f"    Test Recall Churn = "
            f"{test_best['churn_recall']:.3f}"
        )
        print(
            f"    Test F1 Churn = "
            f"{test_best['churn_f1']:.3f}"
        )

        return self

    # ========================================================
    # FEATURE NAMES
    # ========================================================
    def _get_transformed_feature_names(self):
        preprocessor = self.best_model.named_steps[
            "preprocessor"
        ]

        try:
            return [
                str(x)
                for x in (
                    preprocessor.get_feature_names_out()
                )
            ]
        except Exception:
            return None

    # ========================================================
    # 8. SAVE
    # ========================================================
    def save_artifacts(self):
        print(
            "\n[8/8] Menyimpan artefak ..."
        )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Full pipeline includes preprocessing.
        joblib.dump(
            self.best_model,
            MODELS_DIR / "best_model.pkl",
        )

        self.feature_names = (
            self._get_transformed_feature_names()
        )

        joblib.dump(
            self.feature_names,
            MODELS_DIR / "feature_names.pkl",
        )

        # Kept for compatibility with existing app.py.
        joblib.dump(
            None,
            MODELS_DIR / "scaler.pkl",
        )

        joblib.dump(
            None,
            MODELS_DIR / "label_encoders.pkl",
        )

        metadata = {
            "created_at": datetime.now().isoformat(),
            "best_model": self.best_model_name,
            "random_state": self.random_state,
            "test_size": TEST_SIZE,
            "cv_folds": N_SPLITS,
            "target_definition": {
                "1": "Loyal (Output=Yes)",
                "0": "Churn (Output=No)",
            },
            "business_positive_class": (
                "Churn (Target=0)"
            ),
            "business_threshold_rule": {
                "minimum_churn_precision": (
                    MIN_CHURN_PRECISION
                ),
                "threshold_grid": [
                    float(x)
                    for x in THRESHOLD_GRID
                ],
                "selection_priority": [
                    "F1 Churn",
                    "Recall Churn",
                    "Accuracy",
                    "Higher threshold",
                ],
            },
            "model_selection_rule": [
                "CV Recall Churn",
                "CV F1 Churn",
                "CV ROC-AUC",
            ],
            "best_churn_threshold": (
                self.best_churn_threshold
            ),
            "features_used": (
                self.categorical_features
                + self.numerical_features
            ),
            "transformed_feature_names": (
                self.feature_names
            ),
            "features_excluded": {
                "Family size": (
                    "deterministik terhadap "
                    "Customer Type; potensi leakage"
                ),
                "Customer Type": (
                    "bagian dari definisi label"
                ),
                "Pin code / latitude / longitude": (
                    "tidak digunakan berdasarkan "
                    "analisis sebelumnya"
                ),
            },
            "cv_results": self.cv_results,
            "oof_threshold_results": self.oof_results,
            "test_results": self.test_results,
            "dataset_info": {
                "rows": int(len(self.df)),
                "columns": int(
                    len(self.df.columns)
                ),
                "duplicates_detected": int(
                    self.df.duplicated().sum()
                ),
            },
        }

        with open(
            MODELS_DIR / "model_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(
            f"\nArtefak tersimpan di: "
            f"{MODELS_DIR}/"
        )
        print("  - best_model.pkl")
        print("  - feature_names.pkl")
        print("  - scaler.pkl (placeholder)")
        print(
            "  - label_encoders.pkl "
            "(placeholder)"
        )
        print("  - model_metadata.json")

        return self

    # ========================================================
    # PREDICTION UTILITY
    # ========================================================
    def _validate_new_data(self, X_new):
        expected_cols = (
            self.categorical_features
            + self.numerical_features
        )

        missing = [
            c for c in expected_cols
            if c not in X_new.columns
        ]

        if missing:
            raise ValueError(
                f"Kolom input kurang: {missing}"
            )

        X_new = X_new[
            expected_cols
        ].copy()

        for col in self.categorical_features:
            X_new[col] = (
                X_new[col]
                .astype(str)
                .str.strip()
            )

        X_new["Age"] = pd.to_numeric(
            X_new["Age"],
            errors="raise",
        )

        return X_new

    def predict_loyalty_score(
        self,
        X_new: pd.DataFrame,
    ):
        """
        Probability Loyal (Target=1) dalam persen.
        """
        X_new = self._validate_new_data(
            X_new
        )

        proba_loyal = (
            self.best_model
            .predict_proba(X_new)[:, 1]
        )

        return np.round(
            proba_loyal * 100,
            1,
        )

    def predict_churn_probability(
        self,
        X_new: pd.DataFrame,
    ):
        """
        Probability Churn dalam persen.
        """
        X_new = self._validate_new_data(
            X_new
        )

        proba_loyal = (
            self.best_model
            .predict_proba(X_new)[:, 1]
        )

        return np.round(
            (1.0 - proba_loyal) * 100,
            1,
        )

    def predict_churn_flag(
        self,
        X_new: pd.DataFrame,
    ):
        """
        1 = berpotensi Churn
        0 = Loyal

        Threshold berasal dari OOF tuning
        dengan minimum Precision Churn 0.40.
        """
        X_new = self._validate_new_data(
            X_new
        )

        proba_loyal = (
            self.best_model
            .predict_proba(X_new)[:, 1]
        )
        proba_churn = 1.0 - proba_loyal

        return (
            proba_churn
            >= self.best_churn_threshold
        ).astype(int)

    # ========================================================
    # RUN
    # ========================================================
    def run(self):
        print("=" * 72)
        print(
            "LOYALTY / CHURN MODEL TRAINING PIPELINE"
        )
        print("=" * 72)

        (
            self
            .load_data()
            .clean_data()
            .build_target()
            .build_features()
            .split_data()
            .train_models()
            .evaluate_and_select()
            .save_artifacts()
        )

        print("\n" + "=" * 72)
        print("PIPELINE SELESAI")
        print("=" * 72)

        return self


if __name__ == "__main__":
    pipeline = LoyaltyChurnPredictor()
    pipeline.run()
