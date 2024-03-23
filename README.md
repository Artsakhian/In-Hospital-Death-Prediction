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


Preprocessing:
Since data is valuable, we opted out from dropping any variables in our dataset and instead filled in the missing values with the means for numerical data. The already OneHotEncoded data which was categorical in nature was filled in with modes (in our case we filled with the majority class). We have also scaled the data using a standard scaler, which should be fitted in the training data and transformed on the test and train. 


Model:
For model selection we have tried using all of the algorithms already familiar to us. To find the best hyperparameters we have separately used the random search. Finally we have used the stacking classifier to get advantage of all of the models. We have used a Logistic Regression model, Support Vector Classifier, Quadratic Discriminant Analysis, Random Forest Classifier, and Gradient Boosting Classifier. 




Pipeline:
By default the pipeline is going to be run on a training mode. If this is the case, then from the total dataset (which also includes the Target variable) we drop the Target, preprocess the data, fit our model and save both the preprocessor and the model to use for testing (in case the user decided to right away proceed with testing). 
In the case of testing, our solution requires to separately read a file named “y_test” using the add_test_labels method in order to calculate the classification report. In testing mode, first of all the preprocessing is performed from the already saved preprocessor in the training stage and then the model is transformed on the testing set. Finally, the probas together with the threshold are saved in a json file. 
We have carefully decided the choice of a threshold and finally decided to prioritize correctly predicting the class 1 which is accountable for the patient death. This can be explained due to the fact that it is better to predict that a person is likely to die and be wrong than to predict that he/she will live and be wrong.
