import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score, f1_score
import os
import glob
# import data

train_path = "Layer 1/L1_(1200x700)x300/Train/*.csv"
test_path = "Layer 1/L1_(1200x700)x300/Test/*.csv"

pathtrain = []
pathtest = []
train_data = []
test_data = []

for fname in glob.glob(train_path):
    pathtrain.append(fname)
pathtrain = sorted(pathtrain)   

for path in pathtrain:
    df = pd.read_csv(path,header = None)
    train_data.append(df)

for fname in glob.glob(test_path):
    pathtest.append(fname)
pathtest = sorted(pathtest)   

for path in pathtest:
    df = pd.read_csv(path, header = None)
    test_data.append(df)


file_name = ["AtomPairs2DCount","AtomPairs2D","EState", "Extended", "Fingerprinterd", "GraphOnly",
"KlekotaRothCount", "KlekotaRoth", "MACCS", "Pubchem", "SubstructureCount", "Substructure"]



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
scaler = StandardScaler()

### Extreme Gradient Boosting 
import xgboost as xgb

max_depth = np.arange(1, 7, 1)
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
parameters = {
    'max_depth': max_depth,
    'learning_rate': learning_rate
}

 
### Random forest 
max_depth = np.arange(1, 7, 1)
max_features = np.arange(0.2, 0.93, 0.05)
parameters = {
    'max_depth': max_depth,
    'max_features': max_features
}

rf = RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1)

#### SVM
from sklearn import svm

Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
kernel = ["rbf", "linear", "sigmoid", "poly"]
parameters = {'C': Cs, 'gamma' : gammas}

SVM = svm.SVC(random_state=0, probability= True)
### Code score 
def score_model(y_test,y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    accuracy.append(accuracy_score(y_test, y_predicted))
    roc_auc.append(roc_auc_score(y_test, y_predicted))
    f1.append(f1_score(y_test, y_predicted))
    sensitivity.append(cm[0,0] / (cm[0,0]+cm[0,1]))
    specificity.append(cm[1,1] / (cm[1,0]+cm[1,1]))
    MCC.append(matthews_corrcoef(y_test, y_predicted))

### Tune model
#Score: balanced_accuracy, neg_log_loss, f1, accuracy, 

def tune_model(train,test):
    y_trainval = train.iloc[:,0]
    X_trainval = train.drop(train.iloc[:,0], axis = 1)
    y_test = test.iloc[:,0]
    X_test= test.drop(test.iloc[:,0], axis = 1)
    X_trainval = scaler.fit_transform(X_trainval)
    X_test = scaler.transform(X_test)
    rf_grid = GridSearchCV(estimator= SVM, param_grid= parameters,cv=skf.get_n_splits(X_trainval, y_trainval), n_jobs=-1,scoring = 'balanced_accuracy',return_train_score = True, iid = False)
    rf_grid.fit(X_trainval, y_trainval)
    best_params = rf_grid.best_params_
    best_model = rf_grid.best_estimator_
    best_model.fit(X_trainval, y_trainval)
    test_score.append(best_model.score(X_test, y_test)) #test score
    y_predicted = best_model.predict(X_test)
    y_proba_trainval = list(best_model.predict_proba(X_trainval)[:,1])
    y_proba_test = list(best_model.predict_proba(X_test)[:,1])
    y_probability = y_proba_trainval + y_proba_test
    result = rf_grid.cv_results_
    return y_test,y_predicted,result, best_params, y_probability

test_score = []
accuracy = []
roc_auc = []
f1 = []
sensitivity =[]
specificity = []
MCC = []
results = []
best_params =[]
df_pro = pd.DataFrame()

for train, test, name in zip(train_data, test_data, file_name):
    tune_model_result = tune_model(train,test)
    y_test = tune_model_result[0]
    y_predicted = tune_model_result[1]
    results.append(tune_model_result[2])
    best_params.append(tune_model_result[3])
    score_model(y_test,y_predicted)
    df_pro[name] = tune_model_result[4]

df = pd.DataFrame({'Test score':test_score,'Accuracy':accuracy,'ROC_AUC_score':roc_auc
,'F1_score':f1,'Sensitivity':sensitivity,'Specificity':specificity,'MCC':MCC, 'best_params':best_params},index= file_name)
df.to_csv('SVM_Score_balanced_accuracy.csv',header=True)

df_pro.to_csv('SVM_Proba_balanced_accuracy.csv', header= True)

results = pd.DataFrame(results)
results.head()
results = pd.DataFrame(results.T)
results.to_csv('results_SVM_balanced_accuracy.csv',header = file_name)
### Search random_seed
for rf_state, cv_state in zip(range(6),range(6)):
    n = 1
    rf = RandomForestClassifier(n_estimators=200,random_state=rf_state,n_jobs=-1)
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=cv_state)
    cv_rf.append(rf_state)
    kfold_state.append(cv_state)
    for train, test in zip(train_data, test_data):
        result = tune_model(train,test)
        y_test = np.array(result[0])
        y_predicted = result[1]
        score_model(y_test,y_predicted)
        n = n + 1

df = pd.DataFrame({'Test score':test_score,'Valid score':valid_score,'Accuracy':accuracy,'ROC_AUC_score':roc_auc
,'F1_score':f1,'Sensitivity':sensitivity,'Specificity':specificity,'MCC':MCC})
df.to_csv('RF_FullSeedscore.csv',header=True)

### Heat map
scores = rf_grid.cv_results_['mean_test_score'].reshape(len(max_depth), len(max_features))
fig = plt.figure(figsize=(10, 6))
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('max_features')
plt.ylabel('max_depth')
plt.colorbar(orientation='horizontal')
plt.xticks(np.arange(len(max_features)), ["{:.2f}".format(i) for i in max_features])
plt.yticks(np.arange(len(max_depth)), max_depth)
plt.title('Cross-Validation Accuracy')
plt.show()