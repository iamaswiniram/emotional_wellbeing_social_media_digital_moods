## 1. Project Title
**Digital Moods: Classifying Emotional Well-Being from Social Media Usage with Ensemble ML, Deep Learning & Hybrid Fusion**

---

## 2. Application Name & Synopsis

**Application Name:** Digital Moods

### **Executive Summary**
**Digital Moods** is an advanced Machine Learning and Deep Learning powered analytics platform designed to decode the complex relationship between social media usage and emotional well-being. By leveraging a Hybrid Fusion of Ensemble Machine Learning (Random Forest, XGBoost) and Deep Learning (MLP), the system classifies a user's dominant emotional state (Happiness, Anxiety, Sadness, Anger) with high precision based on behavioral metrics.

Beyond classification, the platform offers a **holistic Digital Wellness Ecosystem**. It transforms raw usage data into actionable financial and psychological insights, helping users understand the "Opportunity Cost" of their screen time. With a premium, reactive user interface, it bridges the gap between complex data science and accessible personal development.

### **Key Implemented Features**
1.  **🏠 Reactive Home Dashboard**: A consolidated, real-time command center that instantly updates predictions, confidence scores, and platform health rankings as users adjust their inputs.
2.  **🤖 Hybrid ML & DL Classification**: A robust backend that fuses predictions from multiple advanced models to ensure high-confidence emotional analytics.
3.  **⚖️ Digital Balance Sheet**: A unique financial-modeling approach to screen time, calculating "Digital Assets" (Productive use) vs. "Liabilities" (Unproductive time) to derive a **Digital Net Worth**.
4.  **📈 technical Analysis Hub**: A deep-dive environment for data scientists to explore feature correlations, model metrics (F1-Score, ROC-AUC), and exploratory data analysis (EDA).
5.  **🎨 Premium UI/UX**: A commercially polished interface featuring gradient themes, card-based layouts, and responsive Plotly visualizations designed for C-level presentations.
6.  **📖 User Guide & Documentation**: A dedicated in-app tutorial explaining the project methodology and how to navigate the platform.
7.  **📥 Downloadable PDF Wellness Report**: One-click generation of a professional PDF summary containing predicted emotions, digital persona, usage stats, and tailored wellness tips.
8.  **📖 Storytelling Codebase**: An educational, narrative-driven implementation where code comments explain the significance and "why" behind every logic gate and parameter.
9.  **🏗️ Architectural Transparency**: Real-time layer and parameter summaries for both ML and DL models, revealing the "inner workings" of the Digital Brain.
10. **🔬 Advanced Statistical EDA**: Deep integration of Skewness, Kurtosis, and IQR-based Outlier detection to ensure total data transparency.
11. **📊 The Model Scorecard**: Highly detailed performance reporting for every trained architecture, including Accuracy, Weighted Precision, Recall, F1-Score, and ROC-AUC metrics.
12. **🎭 Confusion Matrix Heatmaps**: Visual and text-based error analysis (The "Mirror of Truth") for every model to reveal exactly where classifications succeed or fail.

### **Technical Architecture**
*   **Frontend**: Streamlit (Reactive State Management, Custom CSS Injection).
*   **Backend**: Python, Scikit-Learn, TensorFlow/Keras.
*   **Data Pipeline**: Automated preprocessing (Scaling, SMOTE, One-Hot Encoding).
*   **Deployment**: Optimized for cloud deployment with minimal footprint.

---

## 3. Project Objectives & Approach

**Problem Statement:**  
Modern social media usage impacts emotional well-being. We aim to **classify a user’s dominant emotion** using **structured usage metrics** (screen time, posts, likes, comments, demographics).

**Approach:**
*   **Primary task:** **Supervised multi-class classification** of `Dominant_Emotion`.
*   **Data modalities:**
    *   **Structured/Behavioral:** `Daily_Usage_Time`, `Posts_Per_Day`, `Likes_Received_Per_Day`, `Comments_Received_Per_Day`, `Messages_Sent_Per_Day`, `Platform`, `Age`, `Gender`.
*   **Models:**
    *   **Ensemble ML:** Random Forest, XGBoost, LightGBM (Tuned).
    *   **Deep Learning:** **Multi-Layer Perceptron (MLP)**.
    *   **Hybrid Fusion:** Weighted Late Fusion of Ensemble and DL models.
*   **Deployment:** **Streamlit** (Inference-only).

---

## 3. Deliverables of the project

**General Approach (Pipeline):**  
Data Ingestion → Preprocessing (Cleaning, Scaling, SMOTE) → Feature Engineering → Benchmarking (Baselines) → Advanced Modeling (Ensemble + DL + Hybrid) → Evaluation → **Streamlit app**.

**Advanced Regularization & Tuning:**
To ensure "Enterprise-Grade" robustness and prevent overfitting, we implement a multi-layered regularization strategy:

#### **Machine Learning (Ensemble) Regularization**
*   **Cost-Complexity Pruning (`ccp_alpha`)**: In Random Forest, we penalize the complexity of trees. This "prunes" overgrown branches that would otherwise capture noise in the training data.
*   **L1/L2 Penalties (`reg_alpha` / `reg_lambda`)**: In XGBoost and LightGBM, we apply weights discipline. L1 (Lasso) encourages sparsity (using only the most important features), while L2 (Ridge) prevents any single feature from having a dominant, unstable influence.
*   **Greedy Pruning (`gamma`)**: Minimum loss reduction required to make a further partition on a leaf node.
*   **Error Penalty (`C`)**: In SVM and Logistic Regression, we tune the inverse regularization strength. A smaller `C` creates a wider margin (more regularization), while a larger `C` aims for higher training accuracy (less regularization).

#### **Deep Learning (Neural Network) Regularization**
*   **Dropout (Strategic Forgetfulness)**: We randomly "deactivate" a percentage of neurons during training (e.g., 20-30%). This prevents "co-adaptation," forcing the network to learn redundant and robust internal representations.
*   **Batch Normalization (Internal Harmonization)**: Normalizing the inputs to each layer stabilizes the learning process and provides a slight regularizing effect by adding noise to the activation values.
*   **Early Stopping (The Wisdom to Stop)**: We monitor the validation loss. If the model starts improving on the training set but declining on the validation set, we stop training immediately and restore the best weights.
*   **Deep Learning:** Batch Normalization, Early Stopping (Patience=5), and Dropout layers.
*   **Ensembles:** L1/L2 Regularization (XGBoost/LightGBM), Cost-Complexity Pruning (Random Forest).
*   **Baselines:** Penalty terms (L2 Ridge) and Soft Margin (C) optimization.
*   **Tuning:** Extensive grid search over 20+ hyperparameters using 3-Fold Cross-Validation.

**Model Questions:**
*   Can complex ensembles outperform simple baselines on usage data?
*   Does combining Deep Learning and Random Forest (Hybrid) reduce variance?

**Model Details & Expected Outcomes:**
*   **Baselines:** Establish performance floor (e.g., Logistic Regression).
*   **Ensemble ML:** Random Forest / XGBoost tuned for max F1-score.
*   **Deep Learning:** Dense Neural Network (MLP) tuned for non-linear patterns.
*   **Hybrid Fusion:** `P_final = α * P_Ensemble + (1-α) * P_DL`.
*   **Evaluation Evidence:**
    *   **Classification:** Macro/micro **Precision, Recall, F1**, **Confusion Matrix**, **ROC-AUC**.
    *   **Explainability:** Feature importances (Ensembles).

---

## 4. Resources

**Dataset Source:**
*   **Name:** Social Media Usage and Emotional Well-Being
*   **Data Splits:** `train.csv`, `val.csv`, `test.csv`.

**Software:**
*   **Language:** Python 3.12+
*   **Core Libraries:**
    *   **Data/EDA:** `pandas`, `numpy`, `seaborn`, `matplotlib`
    *   **ML:** `scikit-learn` (LogReg, SVM, KNN), `xgboost`, `lightgbm`
    *   **DL:** `tensorflow`, `keras-tuner`
    *   **PDF Generation:** `fpdf`
    *   **App:** `streamlit`
*   **Environment:** Virtual Environment (`.venv`), Streamlit Cloud.
*   **Versioning:** GitHub with `.gitignore`.

**Deployment:**
*   GitHub repo structure optimized for size (<50MB).
*   `requirements.txt` minimized for Streamlit Cloud.

---

## 5. Individual Details

*   **Name:** Ramasamy A
*   **Email ID:** ramasamy25@gmail.com


---

## 6. Milestones

| Milestone                | Status / Description                               | Acceptance Criteria             |
| :----------------------- | :------------------------------------------------- | :------------------------------ |
| **1. Define Problem**    | **Completed**: Emotional classification scope.     | Clear target labels.            |
| **2. Data Acquisition**  | **Completed**: Loaded `train/val/test`.            | Files validated.                |
| **3. Preprocessing**     | **Completed**: Scaling, Encoding, SMOTE.           | Balanced classes.               |
| **4. Baselines**         | **Completed**: LogReg, SVM, KNN trained.           | Benchmark accuracy established. |
| **5. Ensemble Training** | **Completed**: RF/XGB/LGBM Tuned with Pruning/L2.  | Accuracy > 95%.                 |
| **6. DL Training**       | **Completed**: MLP with BN/EarlyStopping.          | Accuracy > 90%; Robust.         |
| **7. Hybrid Fusion**     | **Completed**: Ensemble + DL Fusion.               | Comparison valid.               |
| **8. Evaluation**        | **Completed**: ROC, F1, Precision.                 | Metrics CSV generated.          |
| **9. Streamlit App**     | **Completed**: Inference-only UI.                  | Fast inference.                 |
| **10. Documentation**    | **Completed**: Final Report, Storytelling & Guide. | Artifacts delivered.            |

---

## A. System Architecture

### Complete Data Flow & Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│  src/data_loader.py (DataLoader)                                │
│    ├─ load_data() → train.csv, val.csv, test.csv               │
│    └─ Column standardization (Daily_Usage_Time)                 │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            PREPROCESSING & FEATURE ENGINEERING                  │
│  src/preprocessing.py                                           │
│    ├─ FeatureEngineer (Custom Transformer)                     │
│    │   ├─ Engagement_Rate = (Likes+Comments+Messages)/(Posts+1)│
│    │   └─ Social_Activity_Index = Usage_Time + Posts*10        │
│    └─ get_preprocessor() → ColumnTransformer                    │
│        ├─ Numeric: SimpleImputer(median) → StandardScaler      │
│        └─ Categorical: SimpleImputer(constant) → OneHotEncoder  │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
            ┌────────────┴────────────┐
            ↓                         ↓
┌──────────────────────┐  ┌──────────────────────┐
│  MACHINE LEARNING    │  │  DEEP LEARNING       │
│  src/models.py       │  │  src/models.py       │
│  (EnsembleClassifier)│  │  (DeepLearningModel) │
│                      │  │                      │
│  9 Model Types:      │  │  MLP Architecture:   │
│  • Random Forest     │  │  • Input (128)       │
│  • XGBoost           │  │  • Dropout (0.3)     │
│  • LightGBM          │  │  • Hidden (64)       │
│  • Logistic Reg      │  │  • Dropout (0.2)     │
│  • SVM               │  │  • Output (Softmax)  │
│  • KNN               │  │                      │
│  • Decision Tree     │  │  Regularization:     │
│  • Voting Ensemble   │  │  • BatchNorm         │
│  • CatBoost          │  │  • Early Stopping    │
│                      │  │  • Adam Optimizer    │
│  Regularization:     │  │                      │
│  • L1/L2 Penalties   │  │  Output: .h5 (284KB) │
│  • Pruning (ccp)     │  │                      │
│  • Gamma (min loss)  │  │                      │
│                      │  │                      │
│  Output: 9 .pkl      │  │                      │
│  (~37 MB total)      │  │                      │
└──────────┬───────────┘  └──────────┬───────────┘
           ↓                         ↓
           └────────┬────────────────┘
                    ↓
       ┌────────────────────────┐
       │   HYBRID FUSION        │
       │  (HybridFusion         │
       │   Classifier)          │
       │                        │
       │  P_final = 0.6·P_ML +  │
       │            0.4·P_DL    │
       └────────────┬───────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                  AUXILIARY COMPONENTS                           │
│                                                                 │
│  Clustering: kmeans_model.pkl (3 personas)                      │
│  Dimensionality: pca_model.pkl (2D projection)                  │
│  Encoders: label_encoder.pkl, preprocessor.pkl                  │
│  Metrics: 6 CSV files (evaluation, feature importance, etc.)    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              STREAMLIT APPLICATION (app.py - 605 lines)         │
│                                                                 │
│  Load 19 artifacts → User Input → Real-time Processing →       │
│  4 Tabs (Home, Guide, Balance, Technical) → PDF Report         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components
- **6 Source Modules** (1,123 lines total)
- **19 Model Artifacts** (~37 MB)
- **4-Tab Interactive UI** with real-time predictions

---

## B. Streamlit & GitHub Deployment Notes

### Complete Repository Structure

```
Social-Media-Emotional-Well-Being/
│
├── app.py (605 lines)              # Main Streamlit application
│
├── src/                            # Source modules (1,123 lines)
│   ├── data_loader.py (49 lines)   # CSV data loading
│   ├── preprocessing.py (67 lines) # Feature engineering
│   ├── models.py (295 lines)       # ML/DL architectures
│   ├── train_pipeline.py (434 lines) # Training workflow
│   ├── visualization.py            # Plotting utilities
│   └── utils.py (278 lines)        # Streamlit helpers, CSS, PDF
│
├── models/                         # 19 trained artifacts (~37 MB)
│   ├── ML Models (9 files):
│   │   ├─ model_random_forest.pkl  (8.8 MB) ← Primary
│   │   ├─ model_xgboost.pkl        (3.4 MB)
│   │   ├─ model_lightgbm.pkl       (372 KB)
│   │   ├─ model_voting.pkl         (24.5 MB)
│   │   └─ 5 additional models
│   ├── DL Model: model_deep_learning.h5 (284 KB)
│   ├── Preprocessing: preprocessor.pkl, label_encoder.pkl
│   ├── Clustering: kmeans_model.pkl, pca_model.pkl
│   └── Metrics: 6 CSV files (evaluation, feature importance, etc.)
│
├── data/                           # Dataset (excluded from repo)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── notebooks/
│   └── DigitalMoods_Emotional_Wellbeing_from_SocialMedia_Usage.ipynb
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Excludes data/, .venv/
```

### Deployment Guide

**Local Development:**
```bash
# 1. Clone and navigate
git clone <repo-url>
cd Social-Media-Emotional-Well-Being

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
# Opens at http://localhost:8501
```

**Streamlit Cloud Deployment:**
1. Push repository to GitHub
2. Visit share.streamlit.io
3. Connect GitHub repository
4. Select `app.py` as main file
5. Deploy (2-5 minutes)

**File Size Management:**
- All models < 100 MB (GitHub limit)
- Total repo size: ~40 MB (acceptable)
- Optional: Use Git LFS for large files
- Optimization: `@st.cache_resource` for model loading

---

## C. Ethical & Responsible ML

*   **Anonymization:** No PII is stored.
*   **Bias:** SMOTE prevents minority class suppression.
*   **Disclaimer:** “Not a medical diagnostic system.”

---

## E. Acceptance Criteria Checklist

*   [x] Clear problem statement and task framing.
*   [x] Comparison with Baselines (LogReg, SVM, etc.).
*   [x] Sound methodology (Ensemble + DL + Hybrid).
*   [x] Proper evaluation (F1, ROC-AUC, CM).
*   [x] Advanced Regularization (BatchNorm, Pruning, L1/L2).
*   [x] Comprehensive "Multi-Metric" Performance Reporting (Precision, Recall, F1, ROC-AUC).
*   [x] Deployment feasibility (Streamlit).
