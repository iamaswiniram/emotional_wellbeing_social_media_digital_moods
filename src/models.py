import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

class EnsembleClassifier:
    """
    The 'Council of Trees'. This class acts as a coordinator that can summon various 
    machine learning algorithms to analyze our data from different perspectives.
    """
    def __init__(self, model_type='random_forest', **kwargs):
        self.model_type = model_type
        self.model = self._get_model(model_type, **kwargs)

    def _get_model(self, model_type, **kwargs):
        # Depending on our story's needs, we choose different 'Experts':
        if model_type == 'random_forest':
            # The 'Reliable Veteran': Great at handling tabular data with many features.
            return RandomForestClassifier(**kwargs)
        elif model_type == 'xgboost':
            # The 'Overachiever': Uses gradient boosting to iteratively fix errors.
            return XGBClassifier(eval_metric='mlogloss', **kwargs)
        elif model_type == 'lightgbm':
            # The 'Speedster': Highly efficient and optimized for larger datasets.
            return LGBMClassifier(verbose=-1, **kwargs)
        elif model_type == 'logistic_regression':
            # The 'Pure Logic': A baseline that looks for simple linear relationships.
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, **kwargs)
        elif model_type == 'svm':
            # The 'Boundary Maker': Finds the best gap between different emotional states.
            from sklearn.svm import SVC
            return SVC(probability=True, **kwargs) # Probabilities needed for Fusion
        elif model_type == 'knn':
            # The 'Socialite': Classifies users based on similar neighbors.
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**kwargs)
        elif model_type == 'decision_tree':
            # The 'Logical Path': A simpler tree-based model for clear decision paths.
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train):
        # We present the historical evidence to our expert so they can learn.
        self.model.fit(X_train, y_train)

    def tune_hyperparameters(self, X_train, y_train, param_grid, n_iter=20, cv=5, scoring='f1_weighted'):
        """
        The 'Optimization Quest'. We look for the best settings (hyperparameters) 
        to maximize our expert's wisdom.
        """
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        print(f"Tuning {self.model_type} with {n_iter} iterations...")
        
        # We use StratifiedKFold to ensure each fold has the same balance of emotions.
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # We try random combinations of settings. 
        # Using 'f1_weighted' ensures the model cares about performance across all emotions.
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=skf,
            verbose=1,
            random_state=42,
            n_jobs=1 
        )
        search.fit(X_train, y_train)
        print(f"Best Params for {self.model_type}: {search.best_params_}")
        self.model = search.best_estimator_ # The expert is now 'upgraded' with best settings.

    def print_architecture(self):
        """
        Displays the 'Blueprint' of the Ensemble model, showing its core 
        specifications and hyperparameters.
        """
        print(f"--- 🏗️ Architecture & Safeguards (Regularization): {self.model_type.upper()} ---")
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            print(f"Estimators/Units:       {params.get('n_estimators', 'N/A')}")
            print(f"Max Depth:              {params.get('max_depth', 'Unlimited')}")
            
            # --- Printing Regularization Safeguards ---
            print(f"--- Regularization Techniques ---")
            if 'ccp_alpha' in params:
                 print(f"Pruning (ccp_alpha):    {params['ccp_alpha']} (Prevents over-growth)")
            if 'reg_alpha' in params:
                 print(f"L1 Penalty (reg_alpha): {params['reg_alpha']} (Weight Discipline)")
            if 'reg_lambda' in params:
                 print(f"L2 Penalty (reg_lambda):{params['reg_lambda']} (Smoothing weights)")
            if 'gamma' in params and self.model_type != 'svm':
                 print(f"Min Loss Reduction:     {params['gamma']} (Greedy Pruning)")
            if 'C' in params:
                 print(f"Error Penalty (C):      {params['C']} (Inverse Regularization)")
            if 'penalty' in params:
                 print(f"Penalty Type:           {params['penalty']}")
            print(f"---------------------------------")
        else:
            print("Standard non-parametric model.")
        print("-" * 35)

    def predict(self, X_test):
        # Asking the expert for their final verdict on new, unseen cases.
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        # The 'Final Exam': Comparing predictions against reality to see where we stand.
        y_pred = self.predict(X_test)
        print(f"--- Evaluation for {self.model_type} ---")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score (Weighted):", f1_score(y_test, y_pred, average='weighted'))
        return y_pred

class HybridFusionClassifier:
    """
    The 'Supreme Mediator'. This model combines the 'Wisdom of the Ensemble' 
    with the 'Intuition of Deep Learning' to create a final, balanced prediction.
    """
    def __init__(self, ensemble_model, dl_model, alpha=0.6):
        self.ensemble_model = ensemble_model
        self.dl_model = dl_model
        # Alpha is our 'Trust Coefficient'. 0.6 means we give 60% weight to 
        # the Ensemble and 40% to Deep Learning.
        self.alpha = alpha 
        self.classes_ = None
        if hasattr(ensemble_model, 'classes_'):
             self.classes_ = ensemble_model.classes_
        
    def predict_proba(self, X_ml, X_dl=None):
        # We get opinions from both 'Experts' as probability scores.
        if X_dl is None: X_dl = X_ml
            
        p_ml = self.ensemble_model.predict_proba(X_ml)
        p_dl = self.dl_model.predict_proba(X_dl)
        
        # The Fusion: A weighted consensus that reduces the risk of individual bias.
        return self.alpha * p_ml + (1 - self.alpha) * p_dl
        
    def predict(self, X_ml, X_dl=None):
        # Selecting the emotion with the highest consensus probability.
        probs = self.predict_proba(X_ml, X_dl)
        return np.argmax(probs, axis=1)

    def print_architecture(self):
        """
        Displays the 'Fusion Logic' of the Hybrid model.
        """
        print("--- Hybrid Fusion Architecture ---")
        print(f"Core Ensemble Component: {type(self.ensemble_model).__name__}")
        print(f"Deep Learning Component: MLP (Neural Network)")
        print(f"Fusion Weight (Alpha): {self.alpha} (Ensemble) / {1-self.alpha:.1f} (DL)")
        print("-" * 30)

class DeepLearningModel:
    """
    The 'Digital Brain'. An Artificial Neural Network (MLP) designed to 
    find complex, non-linear relationships that simple models might miss.
    """
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_baseline_model()

    def _build_baseline_model(self):
        # Building the initial architecture: Layers of neurons connected together.
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        
        model = Sequential([
            # 128 Neurons: A dense layer to capture broad patterns.
            Dense(128, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3), # 'Forgetfulness' to prevent overfitting.
            
            # 64 Neurons: Refining the insights.
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Final Layer: Outputting probabilities for each emotional class.
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return model

    def build_hypermodel(self, hp):
        """
        The 'Evolutionary Architect'. Allows the model to redesign its own 
        structure during the hyperparameter tuning phase.
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        
        # Input Layer: Testing different sizes (32 to 256 neurons).
        model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=256, step=32), 
                        input_shape=(self.input_dim,)))
        model.add(BatchNormalization()) # Stabilizing the learning process
        model.add(Activation('relu'))
        model.add(Dropout(rate=hp.Float('dropout_input', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Dynamic Hidden Layers: Testing how deep our brain should be (1 to 3 layers).
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
            
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Testing different 'learning speeds' (Learning Rate).
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, max_trials=10, epochs=10):
        # The 'Selection Process': Running 10 different 'Life Cycles' (Trials) 
        # to find the best brain architecture.
        import keras_tuner as kt
        from tensorflow.keras.callbacks import EarlyStopping
        
        print("Starting Keras Tuner...")
        tuner = kt.RandomSearch(
            self.build_hypermodel,
            objective='val_accuracy',
            max_trials=max_trials,
            directory='kt_dir',
            project_name='emotional_wellbeing_kt',
            overwrite=True
        )
        
        # Stop early if the brain stops getting smarter (Patience=3).
        stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1, callbacks=[stop_early])
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.hypermodel.build(best_hps)
        return self.model

    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
        # The 'Final Training': Deeply engraving the knowledge into our neural networks.
        from tensorflow.keras.callbacks import EarlyStopping
        
        # ES prevents 'Memorization' (Overfitting) by stopping when internal error 
        # on unseen validation data starts to rise.
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(X_train, y_train, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 validation_data=validation_data,
                                 callbacks=[es],
                                 verbose=1)
        return history

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return acc

    def print_summary(self):
        """
        Reveals the 'Synaptic Layers' of the Digital Brain, including nodes 
        and connection parameters.
        """
        print("--- 🧠 Deep Learning Architecture & Synaptic Summary ---")
        self.model.summary()
        print("\n--- 🛡️ Safeguards (Regularization Summary) ---")
        print("1. Dropout:  Prevents 'Co-adaptation' by randomly forgetting neurons.")
        print("2. BatchNormalization: Stabilizes training and acts as weak regularization.")
        print("3. EarlyStopping: Wisdom to stop training before overfitting begins.")
        print("-" * 35)
