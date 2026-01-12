from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    The 'Alchemist' of our project. We take raw metrics and transform them into 
    higher-level behavioral insights that reveal deeper emotional patterns.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Story 1: The 'Engagement Echo'. 
        # We calculate how much interaction a user generates relative to their posting 
        # frequency. We add 1 to the denominator to handle those who read but never post.
        X['Engagement_Rate'] = (
            X['Likes_Received_Per_Day'] + 
            X['Comments_Received_Per_Day'] + 
            X['Messages_Sent_Per_Day']
        ) / (X['Posts_Per_Day'] + 1)
        
        # Story 2: The 'Digital Footprint Index'.
        # This combines usage time with posting activity (weighted by 10) to 
        # create a single score representing how active a user is in the digital space.
        X['Social_Activity_Index'] = (
            X['Daily_Usage_Time'] + 
            X['Posts_Per_Day'] * 10 
        )
        return X

def get_preprocessor(numeric_features, categorical_features):
    """
    The 'Refinery'. This function builds the pipeline that cleans and 
    standardizes our features so the models can understand them perfectly.
    """
    # For numbers: We fill missing values with the 'middle' value (median) 
    # and then scale everything so large numbers don't overpower small ones.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Bringing all metrics to a common scale
    ])

    # For categories (like Platform): We mark 'missing' data and then 
    # expand them into binary columns (One-Hot Encoding).
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Pivot categories to flags
    ])

    # Combining everything into a single master processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
