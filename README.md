# 🧠 Digital Moods | Social Media Well-Being Analysis

**Digital Moods** is an enterprise-grade **Machine Learning & Deep Learning** application that explores the intersection of social media usage and emotional well-being using a **Hybrid Fusion** of Ensemble models and Neural Networks.

## 🚀 Key Features

*   **🏠 Reactive Dashboard**: Real-time emotional prediction and confidence scoring.
*   **⚖️ Digital Balance Sheet**: Calculates your "Digital Net Worth" (Assets vs. Liabilities) based on productive usage.
*   **🤖 Hybrid ML/DL Engine**: Combines Random Forest, XGBoost, and MLP (Deep Learning) for robust classification.
*   **📥 PDF Wellness Report**: One-click generation of a downloadable, professional summary with tailored tips.
*   **🎨 Premium UI/UX**: Commercial-grade interface with card-based layouts and responsive Plotly charts.
*   **🔬 Technical Analysis**: Deep dive into feature correlations, model metrics (F1, ROC-AUC), and PCA projections.

## 📂 Repository Structure

```
├── app.py                  # Main Streamlit Application (Inference)
├── models/                 # Pre-trained models (.pkl, .h5) & Metrics
├── src/
│   ├── preprocessing.py    # Feature Engineering & Pipeline Logic
│   ├── models.py           # Model Definitions (Ensemble + DL)
│   ├── train_pipeline.py   # Training Workflow
│   └── utils.py            # Helpers, CSS, PDF Generator
├── data/                   # Dataset (Train/Val/Test)
├── notebooks/              # Master Capstone Notebook (.ipynb)
├── requirements.txt        # Production Dependencies
└── project_proposal.md     # Detailed Technical Documentation
```

## 🛠️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd digital-moods
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 🏗️ Technical Stack

*   **Frontend**: Streamlit, Plotly, HTML/CSS
*   **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
*   **Deep Learning**: TensorFlow/Keras, Keras Tuner
*   **Reporting**: FPDF
*   **Data Processing**: Pandas, NumPy
