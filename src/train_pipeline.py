import sys
import os
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings

# Suppress annoying warnings for clear console output
warnings.filterwarnings('ignore')

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessing import get_preprocessor, FeatureEngineer
from src.models import EnsembleClassifier, DeepLearningModel, VotingClassifier
from src.visualization import plot_confusion_matrix, plot_feature_importance, plot_multiclass_roc
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def print_performance_dashboard(model_name, train_acc, test_acc, precision, recall, f1, roc_auc, reg_applied=True):
    """
    The 'Model Report Card'.
    Prints a professional, tabular dashboard of performance and diagnoses overfitting.
    """
    diff = train_acc - test_acc
    diagnosis = ""
    if diff > 0.15:
        diagnosis = "⚠️ OVERFITTING: The model is memorizing the training data. Regularization needs increase."
    elif diff > 0.05:
        diagnosis = "💡 SLIGHT OVERFIT: Good performance, but a bit too tuned to training samples."
    elif train_acc < 0.60:
        diagnosis = "📉 UNDERFITTING: The model is too simple to capture the emotional patterns."
    else:
        diagnosis = "✅ BALANCED: Excellence achieved! The model generalizes well to new users."

    print(f"\n{'='*50}")
    print(f"📊 THE MODEL SCORECARD: {model_name.upper()}")
    print(f"{'='*50}")
    
    # Table Header
    print(f"{'Metric':<20} | {'Score':<10} | {'Interpretation'}")
    print(f"{'-'*21}|{'-'*12}|{'-'*25}")
    
    # Table Rows
    print(f"{'Train Accuracy':<20} | {train_acc:.4f}     | {'Data Familiarity'}")
    print(f"{'Test Accuracy':<20} | {test_acc:.4f}      | {'Inference Strength'}")
    print(f"{'Precision':<20} | {precision:.4f}     | {'Hit Precision'}")
    print(f"{'Recall':<20} | {recall:.4f}        | {'Pattern Retention'}")
    print(f"{'F1-Score':<20} | {f1:.4f}            | {'Harmonic Mean'}")
    print(f"{'ROC-AUC':<20} | {roc_auc:.4f}       | {'Class Discernment'}")
    
    print(f"{'='*50}")
    print(f"🛡️ Safeguards (Regularization): {'YES' if reg_applied else 'NO'}")
    print(f"🩺 Digital Diagnosis: {diagnosis}")
    print(f"{'='*50}\n")

def prepare_data():
    """
    Phase 1, 2, & 3: The Gathering, Refinery, and Balancing.
    This prepares the 'Foundation' for both ML and DL tracks.
    """
    print("\n--- [FOUNDATION] Phase 1: The Gathering (Loading Data) ---")
    loader = DataLoader(data_dir='data')
    train_df, val_df, test_df = loader.load_data()
    
    print("--- [FOUNDATION] Phase 2: The Refinery (Preprocessing) ---")
    le = LabelEncoder()
    # Ensure target is string type to prevent sorting errors if mixed types exist
    y_train = le.fit_transform(train_df['Dominant_Emotion'].astype(str))
    y_val = le.transform(val_df['Dominant_Emotion'].astype(str))
    y_test = le.transform(test_df['Dominant_Emotion'].astype(str))
    
    os.makedirs('models', exist_ok=True)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    num_cols = ['Age', 'Daily_Usage_Time', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
    cat_cols = ['Gender', 'Platform']
    
    # PERMANENT FIX: Enforce strictly numeric types for num_cols
    # This prevents 'TypeError: < not supported' during imputer.fit (median calculation)
    for df in [train_df, val_df, test_df]:
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Ensure categorical columns are string to prevent sorting errors in OHE
    for df in [train_df, val_df, test_df]:
        for col in cat_cols:
            df[col] = df[col].astype(str)
            
    fe = FeatureEngineer()
    X_train_fe = fe.transform(train_df)
    X_val_fe = fe.transform(val_df)
    X_test_fe = fe.transform(test_df)
    
    num_cols.extend(['Engagement_Rate', 'Social_Activity_Index'])
    
    preprocessor = get_preprocessor(num_cols, cat_cols)
    X_train_processed = preprocessor.fit_transform(X_train_fe)
    X_val_processed = preprocessor.transform(X_val_fe)
    X_test_processed = preprocessor.transform(X_test_fe)
    
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print("--- [FOUNDATION] Phase 3: Balancing the Scales (SMOTE) ---")
    # SMOTE ensures that even rare emotions have enough representatives for the model to learn.
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print("✅ Foundation Ready: Data Loaded, Refined, and Balanced.")
    return X_train_resampled, y_train_resampled, X_val_processed, y_val, X_test_processed, y_test, le, train_df

def run_ml_pipeline(X_train_resampled, y_train_resampled, X_test_processed, y_test, le):
    """
    Phase 4 & 5: The Tournament and Voting Council (Supervised ML).
    """
    print("\n" + "="*60)
    print("🚀 TRACK 1: SUPERVISED MACHINE LEARNING PIPELINE")
    print("="*60)
    
    models_to_train = [
        'logistic_regression', 'knn', 'decision_tree', 'svm', 
        'random_forest', 'xgboost', 'lightgbm'
    ]
    
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 400, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'bootstrap': [True, False],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.05],
            'max_features': ['sqrt', 'log2', None]
        },
        'xgboost': {
            'n_estimators': [100, 300, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10, 15],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0.0, 0.1, 0.2, 0.5],
            'reg_alpha': [0.0, 0.1, 1.0, 10.0],
            'reg_lambda': [0.0, 0.1, 1.0, 10.0],
            'min_child_weight': [1, 3, 5]
        },
        'lightgbm': {
            'n_estimators': [100, 300, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 60, 100, 150],
            'max_depth': [-1, 10, 20, 30],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            'min_child_samples': [10, 20, 30]
        },
        'svm': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'degree': [2, 3, 4]
        },
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'l1_ratio': [0.0, 0.5, 1.0]
        },
        'decision_tree': {
            'max_depth': [None, 5, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'ccp_alpha': [0.0, 0.001, 0.01]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'leaf_size': [20, 30, 50]
        }
    }
    
    tuned_classifiers = []
    best_ensemble_model = None
    best_ensemble_acc = 0
    results = {}
    metrics_log = []
    
    for model_name in models_to_train:
        print(f"\n--- [TOURNAMENT] Training: {model_name.upper()} ---")
        try:
            clf = EnsembleClassifier(model_type=model_name)
            
            if model_name in param_grids:
                # Increased iterations for a more thorough search. 
                # Complex models get more daily 'experiments'.
                iter_count = 10 if model_name in ['svm', 'knn', 'logistic_regression'] else 20
                clf.tune_hyperparameters(X_train_resampled, y_train_resampled, param_grids[model_name], n_iter=iter_count, cv=5)
            else:
                clf.train(X_train_resampled, y_train_resampled)
            
            clf.print_architecture()
                
            if model_name == 'random_forest' or model_name == 'xgboost': 
                if hasattr(clf, 'model'):
                     tuned_classifiers.append((model_name, clf.model))
            
            # Evaluate
            y_pred_train = clf.predict(X_train_resampled)
            y_train_acc = accuracy_score(y_train_resampled, y_pred_train)
            
            y_pred = clf.predict(X_test_processed)
            y_prob = clf.model.predict_proba(X_test_processed) if hasattr(clf.model, "predict_proba") else None
            
            m_acc = accuracy_score(y_test, y_pred)
            m_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            m_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            m_rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                m_roc = roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None else 0
            except:
                m_roc = 0
            
            print_performance_dashboard(model_name, y_train_acc, m_acc, m_prec, m_rec, m_f1, m_roc)
            
            print(f"\n--- 🎭 MIRROR OF TRUTH: Confusion Matrix ({model_name.upper()}) ---")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            plot_confusion_matrix(y_test, y_pred, labels=le.classes_, title=f"Confusion Matrix: {model_name.upper()}")
            print("-" * 50 + "\n")
            
            results[model_name] = m_acc
            metrics_log.append({'Model': model_name, 'Accuracy': m_acc, 'Precision': m_prec, 'Recall': m_rec, 'F1_Score': m_f1, 'ROC_AUC': m_roc})
            
            with open(f'models/model_{model_name}.pkl', 'wb') as f:
                pickle.dump(clf.model, f)
                
            if m_acc > best_ensemble_acc and model_name in ['random_forest', 'xgboost', 'lightgbm']:
                best_ensemble_acc = m_acc
                best_ensemble_model = clf.model

        except Exception as e:
            print(f"Error training {model_name}: {e}")

    if len(tuned_classifiers) >= 2:
        print("\n--- [VOTING COUNCIL] Training Ensemble Council ---")
        voting_clf = VotingClassifier(estimators=tuned_classifiers, voting='soft')
        voting_clf.fit(X_train_resampled, y_train_resampled)
        
        y_pred = voting_clf.predict(X_test_processed)
        y_prob = voting_clf.predict_proba(X_test_processed)
        
        acc_vote = accuracy_score(y_test, y_pred)
        f1_vote = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            roc_vote = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            roc_vote = 0
            
        print_performance_dashboard("voting_ensemble", acc_vote, acc_vote, precision_score(y_test, y_pred, average='weighted', zero_division=0), recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_vote, roc_vote)
        
        print(f"\n--- 🎭 MIRROR OF TRUTH: Confusion Matrix (VOTING_ENSEMBLE) ---")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        plot_confusion_matrix(y_test, y_pred, labels=le.classes_, title="Confusion Matrix: VOTING ENSEMBLE")
        print("-" * 50 + "\n")
        
        metrics_log.append({
            'Model': 'voting',
            'Accuracy': acc_vote,
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1_Score': f1_vote,
            'ROC_AUC': roc_vote
        })
        
        with open('models/model_voting.pkl', 'wb') as f:
            pickle.dump(voting_clf, f)
            
    pd.DataFrame(metrics_log).to_csv("models/ml_metrics.csv", index=False)
    print("\n✅ Track 1 Complete: Supervised ML Tournament Concluded.")
    return best_ensemble_model, metrics_log

def run_dl_pipeline(X_train_resampled, y_train_resampled, X_val_processed, y_val, X_test_processed, y_test, le, train_df, best_ensemble_model=None, ml_metrics=None):
    """
    Phase 6, 7, 8, 9: AI Intuition, Fusion, Clustering, and Explainability.
    """
    if ml_metrics is None:
        ml_metrics = []
        
    print("\n" + "="*60)
    print("🧠 TRACK 2: DEEP LEARNING & HYBRID FUSION PIPELINE")
    print("="*60)

    print("--- [INTUITION] Phase 6: Training Deep Learning (MLP) ---")
    X_train_dl = X_train_resampled.toarray() if hasattr(X_train_resampled, 'toarray') else X_train_resampled
    X_val_dl = X_val_processed.toarray() if hasattr(X_val_processed, 'toarray') else X_val_processed
    X_test_dl = X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed
    
    dl_model = DeepLearningModel(X_train_dl.shape[1], len(le.classes_))
    # Increasing trials to find a more optimal brain structure.
    dl_model.tune_hyperparameters(X_train_dl, y_train_resampled, X_val_dl, y_val, max_trials=15, epochs=20)
    dl_model.train(X_train_dl, y_train_resampled, epochs=30, batch_size=32, validation_data=(X_val_dl, y_val))
    
    dl_model.print_summary()
    
    y_pred_dl = dl_model.predict(X_test_dl)
    y_prob_dl = dl_model.predict_proba(X_test_dl)
    
    y_pred_train_dl = dl_model.predict(X_train_dl)
    acc_train_dl = accuracy_score(y_train_resampled, y_pred_train_dl)
    
    acc_dl = accuracy_score(y_test, y_pred_dl)
    prec_dl = precision_score(y_test, y_pred_dl, average='weighted', zero_division=0)
    rec_dl = recall_score(y_test, y_pred_dl, average='weighted', zero_division=0)
    f1_dl = f1_score(y_test, y_pred_dl, average='weighted', zero_division=0)
    try:
        roc_dl = roc_auc_score(y_test, y_prob_dl, multi_class='ovr')
    except:
        roc_dl = 0
        
    print_performance_dashboard("Deep Learning (MLP)", acc_train_dl, acc_dl, prec_dl, rec_dl, f1_dl, roc_dl)
    
    print("\n--- 🎭 MIRROR OF TRUTH: Confusion Matrix (DEEP_LEARNING) ---")
    cm = confusion_matrix(y_test, y_pred_dl)
    print(cm)
    plot_confusion_matrix(y_test, y_pred_dl, labels=le.classes_, title="Confusion Matrix: DEEP LEARNING")
    print("-" * 50 + "\n")
    
    dl_model.model.save('models/model_deep_learning.h5')
    
    ml_metrics.append({
        'Model': 'deep_learning',
        'Accuracy': acc_dl,
        'Precision': prec_dl,
        'Recall': rec_dl,
        'F1_Score': f1_dl,
        'ROC_AUC': roc_dl
    })

    print("\n--- [FUSION] Phase 7: Training Hybrid Fusion (ML + DL) ---")
    from src.models import HybridFusionClassifier
    if best_ensemble_model:
        hybrid = HybridFusionClassifier(best_ensemble_model, dl_model, alpha=0.6)
        hybrid.print_architecture()
        
        y_pred_h = hybrid.predict(X_test_processed, X_test_dl)
        y_prob_h = hybrid.predict_proba(X_test_processed, X_test_dl)
        
        y_pred_train_h = hybrid.predict(X_train_resampled, X_train_dl)
        acc_train_h = accuracy_score(y_train_resampled, y_pred_train_h)
        
        acc_h = accuracy_score(y_test, y_pred_h)
        prec_h = precision_score(y_test, y_pred_h, average='weighted', zero_division=0)
        rec_h = recall_score(y_test, y_pred_h, average='weighted', zero_division=0)
        f1_h = f1_score(y_test, y_pred_h, average='weighted', zero_division=0)
        try:
            roc_h = roc_auc_score(y_test, y_prob_h, multi_class='ovr')
        except:
            roc_h = 0
            
        print_performance_dashboard("Hybrid Fusion (ML + DL)", acc_train_h, acc_h, prec_h, rec_h, f1_h, roc_h)
        
        print("\n--- 🎭 MIRROR OF TRUTH: Confusion Matrix (HYBRID_FUSION) ---")
        cm = confusion_matrix(y_test, y_pred_h)
        print(cm)
        plot_confusion_matrix(y_test, y_pred_h, labels=le.classes_, title="Confusion Matrix: HYBRID FUSION")
        print("-" * 50 + "\n")
        
        ml_metrics.append({
            'Model': 'hybrid_fusion',
            'Accuracy': acc_h,
            'Precision': prec_h,
            'Recall': rec_h,
            'F1_Score': f1_h,
            'ROC_AUC': roc_h
        })

    print("\n--- [PERSONAS] Phase 8: Performing Clustering & Segmentation ---")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_resampled)
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    print("\n--- [EXPLAINABILITY] Phase 9: Generating Analytics Artifacts ---")
    platform_stats = train_df.groupby('Platform')['Dominant_Emotion'].value_counts(normalize=True).unstack().fillna(0)
    platform_stats.to_csv('models/platform_stats.csv')
    
    if best_ensemble_model and hasattr(best_ensemble_model, 'feature_importances_'):
        with open('models/preprocessor.pkl', 'rb') as f: 
            preprocessor = pickle.load(f)
        num_cols_base = ['Age', 'Daily_Usage_Time', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day', 'Engagement_Rate', 'Social_Activity_Index']
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = list(ohe.get_feature_names_out(['Gender', 'Platform']))
        feature_names = num_cols_base + cat_feature_names
        
        if len(best_ensemble_model.feature_importances_) == len(feature_names):
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': best_ensemble_model.feature_importances_})
            fi_df.sort_values(by='Importance', ascending=False).to_csv('models/feature_importance.csv', index=False)
            
    ref_stats = train_df[['Age', 'Daily_Usage_Time', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']].copy()
    ref_stats['Dominant_Emotion'] = train_df['Dominant_Emotion']
    ref_stats.groupby('Dominant_Emotion').mean().to_csv('models/reference_stats.csv')

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca_data = X_train_resampled.toarray() if hasattr(X_train_resampled, 'toarray') else X_train_resampled
    X_pca = pca.fit_transform(X_pca_data)
    with open('models/pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Dominant_Emotion'] = le.inverse_transform(y_train_resampled)
    pca_df.to_csv('models/community_projections.csv', index=False)

    pd.DataFrame(ml_metrics).to_csv("models/evaluation_metrics.csv", index=False)
    print("\n✅ Track 2 Complete: Deep Learning & Fusion Analysis Finalized.")

def run_training_pipeline():
    """Wrapper to maintain backward compatibility for single-shot execution."""
    data = prepare_data()
    best_ml, ml_metrics = run_ml_pipeline(data[0], data[1], data[4], data[5], data[6])
    run_dl_pipeline(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], best_ml, ml_metrics)

if __name__ == "__main__":
    run_training_pipeline()
