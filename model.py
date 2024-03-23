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

class Model:
   
    def __init__(self):
        base_classifiers = [
        ('logistic_regression', LogisticRegression(solver='newton-cg', penalty=None, max_iter=200, C=10, random_state=42)),
        ('svc', SVC(C=0.5641157902710026, gamma=0.01, kernel='poly', random_state=42)),
        ('qda', QuadraticDiscriminantAnalysis(reg_param=0.005522117123602399)),
        ('rf', RandomForestClassifier(bootstrap=True, max_depth=17, max_features=None, 
                                      min_samples_leaf=1, min_samples_split=17,n_estimators=196)),
        ('gbc', GradientBoostingClassifier(learning_rate=0.2, max_depth=19, min_samples_leaf=15, min_samples_split=3, n_estimators=75)),
        ('log', LogisticRegression(solver='liblinear', penalty='l1', max_iter=200, class_weight='balanced', C=100)),
        ]
        meta_classifier = RandomForestClassifier(random_state=42)
        stacking_classifier = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier
        )
        
        self.model = stacking_classifier

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)