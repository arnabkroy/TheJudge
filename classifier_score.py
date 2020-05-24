import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

def importData(path):
    df = pd.read_csv(path)
    return df

def scoreModel(models,X,y):
    score_df = pd.DataFrame(columns=['Recall','Precision','F1-Score','Accuracy','ROC','LogLoss'])
    for model in models:       
        ll = abs(cross_val_score(model, X, y, scoring='neg_log_loss', cv=10).mean())
        f1 = abs(cross_val_score(model, X, y, scoring='f1', cv=10).mean())
        ac = abs(cross_val_score(model, X, y, scoring='accuracy', cv=10).mean())
        rc = abs(cross_val_score(model, X, y, scoring='recall', cv=10).mean())
        pr = abs(cross_val_score(model, X, y, scoring='precision', cv=10).mean())
        roc = abs(cross_val_score(model, X, y, scoring='roc_auc', cv=10).mean())

        score_dct = {'Model':[str(model)],'Recall':[rc],'Precision':[pr],'F1-Score':[f1],'Accuracy':[ac],'ROC':[roc],'LogLoss':[ll]}
        temp = pd.DataFrame(data = score_dct)
        score_df = pd.concat([score_df,temp])

    return score_df




    