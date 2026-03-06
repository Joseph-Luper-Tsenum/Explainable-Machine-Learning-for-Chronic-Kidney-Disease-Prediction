# CKD-Insight: Explainable Machine Learning for Chronic Kidney Disease Prediction

> An end-to-end machine learning pipeline for chronic kidney disease (CKD) risk prediction using classical ML models with SHAP-based explainability and an interactive dashboard for model exploration and patient-level interpretation.

**BME6938: Medical AI · Project 1 · Group 6 · University of Florida · Spring 2026**

---

## Clinical Context

Chronic Kidney Disease (CKD) affects an estimated 850 million people worldwide and is a leading cause of morbidity and mortality. Early detection is critical because CKD is often asymptomatic in its earliest, most treatable stages. This project builds a transparent, explainable ML framework that predicts CKD risk from routinely collected clinical and laboratory measurements, enabling healthcare professionals to identify at-risk patients during routine screening. The intended beneficiaries include high-risk patients with comorbid diabetes, hypertension, or cardiovascular disease, as well as primary care settings where specialist access is limited.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **97.5%** | 0.980 | 0.980 | 0.980 | **0.999** |
| **XGBoost** | **97.5%** | 0.980 | 0.980 | 0.980 | **0.998** |
| Logistic Regression | 95.0% | 1.000 | 0.920 | 0.958 | 0.994 |
| Decision Tree | 93.8% | 0.979 | 0.920 | 0.948 | 0.981 |
| SVM | 92.5% | 1.000 | 0.880 | 0.936 | 0.995 |

SHAP analysis identified **specific gravity, hemoglobin, serum creatinine, albumin, packed cell volume, and red blood cell count** as the top predictive biomarkers — consistent with established nephrology knowledge.

## Project Structure

```
project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies with versions
├── CDK_dataset.arff                   # Raw UCI CKD dataset
│
├── notebooks/
│   ├── Phase1_CKD_EDA.ipynb           # Phase I: Exploratory Data Analysis
│   ├── Phase2_CKD_Modeling.ipynb      # Phase II: Model Training & Evaluation
│   └── Phase3_CKD_SHAP.ipynb         # Phase III: SHAP Explainability
│
├── app.py                             # Phase IV: Streamlit frontend (CKD-Insight App)
├── backend.py                         # Phase IV: Flask backend (15 API endpoints)
│
├── models/                            # Trained models & preprocessor
│   ├── ckd_random_forest_tuned.joblib
│   ├── ckd_xgboost_tuned.joblib
│   ├── ckd_logistic_regression_tuned.joblib
│   ├── ckd_svm_tuned.joblib
│   ├── ckd_decision_tree_tuned.joblib
│   ├── preprocessor.joblib
│   ├── X_train_processed.npy
│   ├── X_test_processed.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── feature_names.csv
│   ├── rf_shap_values_test.npy
│   ├── xgb_shap_values_test.npy
│   ├── rf_shap_values_train.npy
│   └── xgb_shap_values_train.npy
│
├── data/
│   └── ckd_cleaned.csv                # Cleaned dataset (18 features, 400 patients)
│
├── figures/                           # All generated plots (SVG)
│
└── reports/
    ├── Project1_Group6_Report.pdf     # Final project report
    ├── CDK_EDA_Report.docx            # Phase I report
    ├── CKD_Modeling_Report.docx       # Phase II report
    ├── CKD_SHAP_Report.docx           # Phase III report
    └── CKD_Insight_Report.docx        # Phase IV report
```

## Dataset

- **Source:** [UCI Machine Learning Repository — Chronic Kidney Disease](https://doi.org/10.24432/C5G020) (Rubini et al., 2015)
- **Format:** ARFF (Attribute-Relation File Format)
- **Size:** 400 patients × 26 columns (24 features + id + target)
- **Target:** Binary classification — `ckd` (250, 62.5%) vs. `notckd` (150, 37.5%)
- **Missing Data:** 1,012 cells across 242/400 rows (non-random pattern)
- **Features retained after leakage removal:** 18 (14 numeric + 4 categorical)
- **Leakage features dropped:** `id`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane` (post-diagnosis clinical flags)
- **License:** CC BY 4.0

## Environment Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
https://github.com/Joseph-Luper-Tsenum/ckd-insight.git
cd ckd-insight

# Install dependencies
pip install -r requirements.txt
```

**Expected install time:** ~2 minutes on a standard machine.

## Quick Start

### Option A: Run the Notebooks (Phases I–III)

```bash
# Phase I: Exploratory Data Analysis
jupyter notebook notebooks/Phase1_CKD_EDA.ipynb

# Phase II: Model Training & Evaluation
jupyter notebook notebooks/Phase2_CKD_Modeling.ipynb

# Phase III: SHAP Explainability
jupyter notebook notebooks/Phase3_CKD_SHAP.ipynb
```

Each notebook runs top-to-bottom. Place `CDK_dataset.arff` in the project root before running Phase I. Phase II and III depend on artifacts from the previous phase.

**Expected runtime:** Phase I ~1 min, Phase II ~5 min (GridSearchCV), Phase III ~2 min.

### Option B: Launch the CKD-Insight App (Phase IV)

```bash
# Terminal 1 — Start Flask backend
python backend.py
# Output: CKD Backend — http://127.0.0.1:5000

# Terminal 2 — Start Streamlit frontend
streamlit run app.py
# Output: App opens at http://localhost:8501
```

**Expected startup time:** ~10 seconds (model loading).

## Usage Guide

### CKD-Insight App Tabs

1. **🔬 Patient Prediction** — Enter 18 clinical features → get CKD probability + SHAP waterfall explanation. Use "Load Sample CKD/Healthy Patient" buttons for quick demo.
2. **📊 Model Performance** — View metrics, confusion matrices, ROC curves, and feature importance for all 5 models.
3. **🌍 SHAP Global** — Explore beeswarm plots, bar plots, and RF vs. XGBoost SHAP comparison.
4. **🔍 SHAP Local** — Browse 80 test patients by TP/TN/FN/FP → per-patient SHAP waterfall. Examine misclassified cases.
5. **📈 SHAP Dependence** — Feature vs. SHAP scatter plots revealing non-linear effects and clinical thresholds.
6. **🗂 Dataset Explorer** — Browse data, feature distributions, missing data summary.
7. **ℹ About** — Project pipeline, results, and author information.

### Notebook Pipeline

| Phase | Notebook | Key Outputs |
|-------|----------|-------------|
| I | `Phase1_CKD_EDA.ipynb` | Class distribution, missingness analysis, statistical tests, leakage investigation, cleaned CSV |
| II | `Phase2_CKD_Modeling.ipynb` | 5 tuned models (.joblib), confusion matrices, ROC curves, feature importance |
| III | `Phase3_CKD_SHAP.ipynb` | SHAP beeswarm, waterfall, force, dependence plots; SHAP value arrays (.npy) |

## Computational Requirements

- **Hardware:** Standard laptop (no GPU required)
- **RAM:** 4 GB minimum
- **OS:** macOS, Linux, or Windows
- **Python:** 3.10+

## Project Pipeline

```
Phase I: EDA                Phase II: Modeling           Phase III: SHAP             Phase IV: App
┌──────────────┐           ┌──────────────────┐        ┌─────────────────┐        ┌───────────────┐
│ Data cleaning │           │ 5 ML models       │        │ TreeExplainer   │        │ Flask backend │
│ Missingness   │──────────►│ GridSearchCV       │───────►│ Global + Local  │───────►│ Streamlit UI  │
│ Leakage check │           │ Cross-validation   │        │ Dependence      │        │ 7 tabs        │
│ Stats tests   │           │ Evaluation         │        │ Comparison      │        │ 15 endpoints  │
└──────────────┘           └──────────────────┘        └─────────────────┘        └───────────────┘
```

## Authors

**Joseph Luper Tsenum**:
Ph.D. Researcher in Biomedical Engineering (Modeling & Biomedical Data Science Specialization), University of Florida.
Joseph develops Generative AI platforms for designing novel oligonucleotides and applies machine learning methods
to biomedical data analysis and drug discovery.

**Riley Bendure**:
M.S. Researcher in Biomedical Engineering (Brain Signal Processing), University of Florida.
Riley utilizes machine learning methods improving modulation of non-motor symptoms in Parkinson's for adaptive deep brain stimulation with previous experience in Cochlear implant temporal signal processing. His aim is to bridge gaps between patient perception and effective neuromodulation in implantable neurostimulators.

**Gopal Viraj Koundinya V. Vutukuru**:
M.S. Student in Biomedical Engineering, University of Florida.
Gopal Viraj is a first‑year M.S. student in Biomedical Engineering at the University of Florida with a strong interest in biomaterials and regenerative medicine. He is open to pursuing both industry work and academic research in these areas in the future. His overall aim is to work in the healthcare industry to solve day‑to‑day diagnostic problems by applying the latest technologies in biomedical engineering.

## Contact Us

Joseph Luper Tsenum: josephtsenum@gatech.edu

Riley Bendure: r.bendure@ufl.edu

Gopal Viraj Koundinya V. Vutukuru: gv.vutukuru@ufl.edu

Address: Malachowsky Hall

1889 Museum Rd, Gainesville, FL 32611, United States


## Citation

If you use this work, please cite:

```bibtex
@misc{tsenum2025ckdinsight,
  title={CKD-Insight: Explainable Machine Learning for Chronic Kidney Disease Prediction},
  author={Tsenum, Joseph Luper and Bendure, Riley and Vutukuru, Gopal Viraj Koundinya V.},
  year={2026},
  institution={University of Florida},
  note={BME6938 Medical AI, Project 1}
}
```
## Bibliography: 

- E. M. Chouit, M. Rachdi, M. Bellafkih, and B. Raouyane, “Interpretable machine learning for chronic kidney disease prediction: Insights from SHAP and LIME analyses,” PLoS One, vol. 21, no. 2, Art. no. e0343205, Feb. 2026.

- I. Balikci Cicek and Z. Kucukakcali, “Explainable Artificial Intelligence Method SHAP’s Prediction of Risk Factors Associated with Chronic Kidney Disease Combined with Black Box Methods,” J. Comm. Med. and Pub. Health Rep., vol. 4, no. 10, Nov. 2023.

- M. A. Islam, M. Z. H. Majumder, and M. A. Hussein, “Chronic kidney disease prediction based on machine learning algorithms,” J. Pathol. Inform., vol. 14, Art. no. 100189, Jan. 2023.

- S. Sharma et al., “Machine Learning Algorithm for Detecting and Predicting Chronic Kidney Disease,” Biomed. & Pharmacol. J., vol. 18, no. 2, pp. 1230–1245, June 2025.

- E. M. Senan et al., “Diagnosis of Chronic Kidney Disease Using Effective Classification Algorithms and Recursive Feature Elimination Techniques,” J. Healthc. Eng., vol. 2021, Art. no. 1004767, June 2021.

- R. K. Halder et al., “ML-CKDP: Machine learning-based chronic kidney disease prediction with smart web application,” J. Pathol. Inform., vol. 15, Art. no. 100371, Feb. 2024.

- B. Metherall, A. K. Berryman, and G. S. Brennan, “Machine learning for classifying chronic kidney disease and predicting creatinine levels using at-home measurements,” Sci. Rep., vol. 15, Art. no. 4330, Feb. 2025.

- P. B. Mark et al., “Global, regional, and national burden of chronic kidney disease in adults, 1990–2023,” The Lancet, vol. 406, no. 10518, pp. 2461–2482, 2025.

- L. Rubini, P. Soundarapandian, and P. Eswaran, “Chronic Kidney Disease,” UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5G020

- S. M. Lundberg and S. I. Lee, “A Unified Approach to Interpreting Model Predictions,” in Proc. NeurIPS, 2017.

- S. M. Lundberg et al., “From local explanations to global understanding with explainable AI for trees,” Nature Machine Intelligence, vol. 2, pp. 56–67, 2020.

- F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” J. Machine Learning Res., vol. 12, pp. 2825–2830, 2011.

- T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proc. KDD, 2016.

- L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

- A. S. Levey and J. Coresh, “Chronic kidney disease,” The Lancet, vol. 379, no. 9811, pp. 165–180, 2012.

- L. S. Shapley, “A value for n-person games,” Contributions to the Theory of Games, vol. 2, no. 28, pp. 307–317, 1953.

- O. Troyanskaya et al., “Missing value estimation methods for DNA microarrays,” Bioinformatics, vol. 17, no. 6, pp. 520–525, 2001.


## License

This project is for educational purposes. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

⚠ **Disclaimer:** This tool is for educational and research purposes only. It is not intended for clinical use or medical decision-making.
