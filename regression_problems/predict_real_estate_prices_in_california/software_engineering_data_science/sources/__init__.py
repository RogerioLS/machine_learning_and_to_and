# sources/__init__.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Importando classes dos módulos individuais
from .a_fetch_dataset import DataFetcher
from .b_clean_dataset import DataCleaner
from .c_feature_engineering import FeatureEngineer
from .d_feature_selection import FeatureSelection
from .e_train_model import ModelTrainer
from .f_generate_evaluation_metrics import ModelEvaluator

__all__ = [
    'pd',
    'fetch_california_housing',
    'StandardScaler',
    'RandomForestRegressor',
    'train_test_split',
    'mean_absolute_error',
    'r2_score',
    'DataFetcher',
    'DataCleaner',
    'FeatureEngineer',
    'FeatureSelection',
    'ModelTrainer',
    'ModelEvaluator'
]