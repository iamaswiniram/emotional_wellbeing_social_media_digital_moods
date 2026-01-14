# Digital Moods | Social Media Emotional Well-Being Analysis

**Digital Moods** is an advanced Machine Learning and Deep Learning platform that classifies emotional well-being states from social media usage patterns. Using a Hybrid Fusion architecture combining Ensemble ML and Neural Networks, the system achieves 95%+ accuracy in predicting dominant emotions (Happiness, Anxiety, Sadness, Anger).
 **Streamlit URL:** : https://emotional-wellbeing-social-media-usage-iamaswiniram.streamlit.app/
---

## Features

### Core Capabilities
- **Real-time Emotion Prediction**: Instant classification with confidence scoring
- **Hybrid ML/DL Engine**: Combines 9 ML models (Random Forest, XGBoost, LightGBM, etc.) with Deep Learning (MLP)
- **User Persona Clustering**: KMeans-based segmentation (Balanced/Passive/Power User)
- **Digital Balance Sheet**: Financial modeling of screen time (Assets vs. Liabilities)
- **Interactive Visualizations**: Platform health rankings, PCA projections, correlation heatmaps
- **PDF Wellness Reports**: Downloadable professional summaries with personalized recommendations

### Application Tabs
1. **Home Dashboard**: Prediction results, confidence gauge, persona insights, platform comparisons
2. **User Guide**: Methodology documentation and usage instructions
3. **Digital Balance**: Opportunity cost analysis and productivity metrics
4. **Technical Analysis**: EDA, model performance metrics, feature importance, ROC curves

---

## Project Structure

```
Social-Media-Emotional-Well-Being/
│
├── app.py (605 lines)              # Main Streamlit application
│
├── src/                            # Source code modules (1,123 lines)
│   ├── data_loader.py              # CSV data loading and validation
│   ├── preprocessing.py            # Feature engineering and transformers
│   ├── models.py                   # ML/DL model architectures
│   ├── train_pipeline.py           # Complete training workflow
│   ├── visualization.py            # Plotting utilities
│   └── utils.py                    # Streamlit helpers, CSS, PDF generation
│
├── models/                         # Trained artifacts (19 files, ~37 MB)
│   ├── model_random_forest.pkl     # Primary model (8.8 MB)
│   ├── model_xgboost.pkl           # Gradient boosting (3.4 MB)
│   ├── model_lightgbm.pkl          # Efficient boosting (372 KB)
│   ├── model_voting.pkl            # Ensemble of ensembles (24.5 MB)
│   ├── model_deep_learning.h5      # Keras MLP (284 KB)
│   ├── model_*.pkl                 # Additional ML models (5 files)
│   ├── preprocessor.pkl            # Fitted ColumnTransformer
│   ├── label_encoder.pkl           # Emotion encoder
│   ├── kmeans_model.pkl            # Clustering model
│   ├── pca_model.pkl               # Dimensionality reduction
│   └── *.csv                       # Metrics and reference statistics (6 files)
│
├── data/                           # Dataset (excluded from repo)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── notebooks/                      # Jupyter notebooks
│   └── DigitalMoods_Emotional_Wellbeing_from_SocialMedia_Usage.ipynb
│
├── requirements.txt                # Python dependencies
├── project_proposal.md             # Detailed technical documentation
└── README.md                       # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Social-Media-Emotional-Well-Being
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## Technical Stack

### Machine Learning
- **Frameworks**: scikit-learn, XGBoost, LightGBM
- **Models**: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, KNN, Decision Tree, Voting Ensemble
- **Regularization**: L1/L2 penalties, cost-complexity pruning, gamma (min loss reduction)

### Deep Learning
- **Framework**: TensorFlow/Keras
- **Architecture**: Multi-Layer Perceptron (128→64→4 neurons)
- **Regularization**: Dropout (0.3, 0.2), Batch Normalization, Early Stopping
- **Tuning**: Keras Tuner for hyperparameter optimization

### Data Processing
- **Libraries**: pandas, numpy, imbalanced-learn
- **Preprocessing**: StandardScaler, OneHotEncoder, ColumnTransformer
- **Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Engineering**: Engagement_Rate, Social_Activity_Index

### Visualization & UI
- **Frontend**: Streamlit with custom CSS
- **Charts**: Plotly (interactive), Seaborn, Matplotlib
- **Reporting**: FPDF for PDF generation

---

## Model Architecture

### Hybrid Fusion Approach
```
User Input → Feature Engineering → Preprocessing
                                        ↓
                    ┌──────────────────┴──────────────────┐
                    ↓                                     ↓
            ML Track (9 models)                   DL Track (MLP)
         Regularized: L1/L2/Pruning            Regularized: Dropout/BN
                    ↓                                     ↓
            Probabilities (P_ML)                  Probabilities (P_DL)
                    └──────────────────┬──────────────────┘
                                       ↓
                          Hybrid Fusion Layer
                    P_final = 0.6·P_ML + 0.4·P_DL
                                       ↓
                          Predicted Emotion + Confidence
```

### Data Pipeline
1. **Ingestion**: Load train/val/test CSV files
2. **Feature Engineering**: Create derived features (Engagement_Rate, Social_Activity_Index)
3. **Preprocessing**: Imputation, scaling, one-hot encoding
4. **Balancing**: SMOTE for class imbalance
5. **Training**: Parallel ML and DL tracks with hyperparameter tuning
6. **Evaluation**: Multi-metric assessment (Accuracy, F1, Precision, Recall, ROC-AUC)
7. **Deployment**: Save 19 artifacts for inference

---

## Performance Metrics

| Model               | Accuracy | F1-Score | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| Random Forest       | 95%+     | 0.94+    | 0.96+   |
| XGBoost             | 94%+     | 0.93+    | 0.95+   |
| LightGBM            | 93%+     | 0.92+    | 0.94+   |
| Deep Learning (MLP) | 92%+     | 0.91+    | 0.93+   |
| Hybrid Fusion       | 95%+     | 0.94+    | 0.96+   |

*Metrics are weighted averages across all emotion classes*

---

## Usage

### Running the Streamlit App

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Configure your profile** (sidebar):
   - Demographics: Age, Gender, Platform
   - Usage metrics: Screen Time, Posts, Likes, Messages

3. **View results**:
   - Predicted emotion and confidence score
   - User persona classification
   - Platform health rankings
   - Digital balance analysis

4. **Download report**:
   - Click "Download Wellness Report (PDF)" for a comprehensive summary

### Training Models (Optional)

To retrain models from scratch:

```bash
python src/train_pipeline.py
```

This will:
- Load and preprocess data
- Train 9 ML models with hyperparameter tuning
- Train Deep Learning MLP with Keras Tuner
- Generate evaluation metrics
- Save all 19 artifacts to `models/` directory

---

## Dependencies

### Core Requirements
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
tensorflow>=2.15.0
streamlit>=1.28.0
plotly>=5.17.0
imbalanced-learn>=0.11.0
fpdf>=1.7.2
keras-tuner>=1.4.0
```

See `requirements.txt` for complete list.

---

## Deployment

### Local Development
Already covered in [Installation & Setup](#installation--setup)

### Streamlit Cloud

1. Push repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy

**Note**: Ensure all model files are included in the repository (total ~37 MB is within GitHub limits).

### Performance Optimization
- Models are cached using `@st.cache_resource` for fast loading
- Training data cached with `@st.cache_data`
- CPU-optimized inference (no GPU required)

---

## Project Documentation

- **Jupyter Notebook**: `DigitalMoods_Emotional_Wellbeing_from_SocialMedia_Usage.ipynb`
  - Complete project walkthrough
  - Detailed architecture diagrams
  - Deployment guides
  - Code implementation

- **Project Proposal**: `project_proposal.md`
  - Technical specifications
  - Methodology details
  - Evaluation criteria

---

## Author

**Ramasamy A**  
Email: ramasamy25@gmail.com  
Batch: 11

---

## License

This project is developed as part of a capstone project for educational purposes.

---

## Acknowledgments

- Dataset: Social Media Usage and Emotional Well-Being
- Frameworks: scikit-learn, TensorFlow, Streamlit
- Inspiration: Intersection of digital wellness and machine learning

