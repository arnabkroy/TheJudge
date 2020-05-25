import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def importData(path):
    df = pd.read_csv(path)
    return df

def scoreModel(models,X,y):
    score_df = pd.DataFrame(columns=['Recall','Precision','F1-Score','Accuracy','ROC','LogLoss'])
    for model in models:
        pos = str(model).find('(')
        name = str(model)[:pos]       
        ll = abs(cross_val_score(model, X, y, scoring='neg_log_loss', cv=10).mean())
        f1 = abs(cross_val_score(model, X, y, scoring='f1', cv=10).mean())
        ac = abs(cross_val_score(model, X, y, scoring='accuracy', cv=10).mean())
        rc = abs(cross_val_score(model, X, y, scoring='recall', cv=10).mean())
        pr = abs(cross_val_score(model, X, y, scoring='precision', cv=10).mean())
        roc = abs(cross_val_score(model, X, y, scoring='roc_auc', cv=10).mean())

        score_dct = {'Model':[name],'Recall':[rc],'Precision':[pr],'F1-Score':[f1],'Accuracy':[ac],'ROC':[roc],'LogLoss':[ll]}
        temp = pd.DataFrame(data = score_dct)
        score_df = pd.concat([score_df,temp])

    return score_df

def plotScores(score_df):
    plt.plot(score_df['Model'],score_df['Recall'],color='orange',marker='o')
    plt.plot(score_df['Model'],score_df['Precision'],color='blue',marker='o')
    plt.plot(score_df['Model'],score_df['F1-Score'],color='green',marker='o')
    plt.plot(score_df['Model'],score_df['Accuracy'],color='magenta',marker='o')
    plt.plot(score_df['Model'],score_df['ROC'],color='red',marker='o')
    plt.plot(score_df['Model'],score_df['LogLoss'],color='grey',marker='o')
    plt.legend(['Recall','Precision','F1-Score','Accuracy','ROC','LogLoss'])
    plt.ylabel('Score')
    plt.xlabel('Models')
    plt.show()

def ChooseModel(score_df,metrics):
    model_dct = {}
    for metric in metrics:
        temp = score_df.copy()
        if metric is 'LogLoss':
            order = True
        else:
            order = False

        temp.sort_values(by=metric,ascending=order,inplace=True)
        model_select = temp['Model'].values[:3]
        model_dct.update({metric:model_select})


    return model_dct

    