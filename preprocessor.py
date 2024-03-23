import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from joblib import load
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        X = X.drop(columns=['Length_of_stay', 'SAPS-I', 'SOFA', 'Survival', 'recordid'])
        X['Gender'] = X['Gender'].fillna(X['Gender'].mean() >= 0.5)
        X['MechVentLast8Hour'] = X['MechVentLast8Hour'].fillna(X['MechVentLast8Hour'].mean() >= 0.5)
        X.fillna(X.mean(), inplace=True)
        self.scaler.fit(X)

    def transform(self, X):
        X = X.drop(columns=['Length_of_stay', 'SAPS-I', 'SOFA', 'Survival', 'recordid'])
        X['Gender'] = X['Gender'].fillna(X['Gender'].mean() >= 0.5)
        X['MechVentLast8Hour'] = X['MechVentLast8Hour'].fillna(X['MechVentLast8Hour'].mean() >= 0.5)
        X.fillna(X.mean(), inplace=True)
        return self.scaler.transform(X)
