import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve , log_loss,roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV

def importData(path):
    df = pd.read_csv(path)
    return df

def scoreModel(model,X,y):       
    cv_score_ll = abs(cross_val_score(model, X, y, scoring='neg_log_loss', cv=10).mean())
    cv_score_f1 = abs(cross_val_score(model, X, y, scoring='f1', cv=10).mean())
    cv_score_ac = abs(cross_val_score(model, X, y, scoring='accuracy', cv=10).mean())
    cv_score_rc = abs(cross_val_score(model, X, y, scoring='recall', cv=10).mean())
    cv_score_pr = abs(cross_val_score(model, X, y, scoring='precision', cv=10).mean())
    cv_score_roc = abs(cross_val_score(model, X, y, scoring='roc_auc', cv=10).mean())

    return cv_score_ll, cv_score_f1, cv_score_ac, cv_score_rc, cv_score_pr, cv_score_roc

if __name__ == "main":

    path = "D:\\DataScience\\MachineHack_FinRisk\\Financial_Risk_Participants_Data\\Financial_Risk_Participants_Data\\Train.csv"
    df = importData(path)
    print(df.head())
    print('Hello')



    