import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from functools import partial
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

tqdm.__init__ = partial(tqdm.__init__, disable=True)
base_dir = Path(__file__).resolve().parent

df = pd.read_csv(f'{base_dir.parent}\\dataset\\dataset_integrato.csv', sep=',')

X = df.drop(columns=['event_death','ID','surv_1yr','surv_2yr','surv_3yr','surv_4yr','surv_5yr'])
X = pd.get_dummies(X, columns=['tnm8', 'region', 'smoking', 'keck', 'dececco'])
X = pd.get_dummies(X, columns=['chemo', 'surg', 'radio', 'sex'], drop_first = True)

y = df['surv_1yr']
auc_scores = []
f1_1 = []
f1_0 = []
f1_com = []

for iteration in range(1, 11):    
    print(f"Starting iteration {iteration}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    rfc = RandomForestClassifier(class_weight='balanced')

    param_distributions = {
        'n_estimators': [120], 
        'max_depth': [6, 8, 12], 
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 2, 5],
        'max_features': [None]
    }

    loo = LeaveOneOut()
    
    grid_search = GridSearchCV(
    estimator=rfc, 
    param_grid=param_distributions,         
    cv=loo,   
    scoring='balanced_accuracy', 
    n_jobs=-1
    )    
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)

    rfc = grid_search.best_estimator_
    rfc.set_params(n_estimators=200)
    rfc.set_params(n_jobs=-1)
    rfc.fit(X_train, y_train)
    y_probs = rfc.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.7).astype(int) #more class balancing needed

    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_probs)

    print(confusion_matrix(y_test, y_pred))
    
    f1_1.append(report['1']['f1-score']) 
    f1_0.append(report['0']['f1-score'])
    f1_com.append(report['weighted avg']['f1-score'])
    auc_scores.append(auc_score)

metrics_dict = {
    'mean': [np.mean(auc_scores), np.mean(f1_0), np.mean(f1_1), np.mean(f1_com)],
    'std_dev': [np.std(auc_scores), np.std(f1_0), np.std(f1_1), np.std(f1_com)]
}
rows = ['auc', 'f1_0', 'f1_1', 'f1_com']
df_metrics = pd.DataFrame(metrics_dict, index=rows)
df_metrics = df_metrics.round(2)
print(df_metrics)
