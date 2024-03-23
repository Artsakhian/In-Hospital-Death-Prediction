import argparse

from preprocessor import Preprocessor
from model import Model

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


class Pipeline:
    def __init__(self):
        self.predictions = None
        self.probas = None
        self.y_test = None
        self.threshold = 0.4
    
    def run(self, data_path, test=False):
        data = pd.read_csv(data_path)

        if test:
            preprocessor = load('preprocessor.joblib')
            model = load('model.joblib')
            df = preprocessor.transform(data)
            self.probas = model.predict_proba(df)
            prob = self.probas.tolist()
            self.probas_to_json(prob, self.threshold)
            print(self.report())
            
        else:
            preprocessor = Preprocessor()
            model = Model()
            
            X = data.drop(['In-hospital_death'], axis=1)
            Target = data['In-hospital_death']
            
            preprocessor.fit(X)
            X = preprocessor.transform(X)
            model.fit(X, Target)

            dump(model, 'model.joblib')
            dump(preprocessor, 'preprocessor.joblib')

    def report(self, treshold = 0.4):
        if self.y_test is not None:
            self.predictions = self.probas[:, 1] >= treshold
            return classification_report(self.y_test, self.predictions)
        else:
            raise ValueError("No test labels given")
        
    def add_test_labels(self, y_train_path):
        self.y_test = pd.read_csv(y_train_path)
        
    def probas_to_json(self, prob, threshold):
        data_to_save = {
            'proba_matrix': prob,
            'threshold': threshold
        }
        json_file_path = 'probas_and_threshold.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(data_to_save, json_file)
            
            
def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to train or test dataset")
    parser.add_argument("--test", action="store_true", help="Run in testing mode")

    args = parser.parse_args()
    
    pipeline = Pipeline()
    
    if args.test:
        pipeline.add_test_labels("y_test.csv")
        pipeline.run(data_path=args.data_path, test=True)
    else:
        pipeline.run(data_path=args.data_path)


if __name__ == "__main__":
    main()