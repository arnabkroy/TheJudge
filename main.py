from classifier_score import importData, scoreModel, plotScores
from classifier_score import scoreModel
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

path = 'D:\\DataScience\\MachineHack_FinRisk\\Financial_Risk_Participants_Data\\Financial_Risk_Participants_Data\\Train.csv'
df = importData(path)

var = ['Location_Score','Internal_Audit_Score','External_Audit_Score','Fin_Score']
X = df[var]
y = df['IsUnderRisk']
classifiers = [LogisticRegression(random_state=100),RandomForestClassifier(random_state=100),GaussianNB(),AdaBoostClassifier(random_state=100),GradientBoostingClassifier(random_state=100)]
score_df = scoreModel(classifiers,X,y)

print(score_df)
plotScores(score_df)