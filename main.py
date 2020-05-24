from classifier_score import importData
from classifier_score import scoreModel
from sklearn.naive_bayes import GaussianNB

path = 'D:\\DataScience\\MachineHack_FinRisk\\Financial_Risk_Participants_Data\\Financial_Risk_Participants_Data\\Train.csv'
df = importData(path)

var = ['Location_Score','Internal_Audit_Score','External_Audit_Score','Fin_Score']
X = df[var]
y = df['IsUnderRisk']
model = GaussianNB()
score_df = scoreModel([model],X,y)

print(score_df)