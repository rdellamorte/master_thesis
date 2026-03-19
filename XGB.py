import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import  GridSearchCV, train_test_split, LeaveOneOut
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm
from functools import partial
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

tqdm.__init__ = partial(tqdm.__init__, disable=True)
base_dir = Path(__file__).resolve().parent

df = pd.read_csv(f'{base_dir.parent}\\dataset\\dataset_integrato.csv', sep=',')

y = df['surv_1yr']
X = df.drop(columns=['event_death','ID','surv_1yr','surv_2yr','surv_3yr','surv_4yr','surv_5yr'])

cat_cols = X.columns.difference(['age'])
X[cat_cols] = X[cat_cols].astype('category')

auc_scores = []
f1_1 = []
f1_0 = []
f1_com = []

for iteration in range(1,11):
    print(f"Starting iteration {iteration}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        eval_metric='logloss', 
        tree_method='hist',         #'exact' requires OHE
        scale_pos_weight=ratio,     #handle class imbalance
        enable_categorical=True     #alternative to OHE
        )
    loo = LeaveOneOut()
    
    param_distributions = {
        'n_estimators': [120],       
        'learning_rate': [0.01, 0.01, 0.1, 0.5],    
        'max_depth': [6, 8, 12],            
        'min_child_weight': [1, 5],
        'gamma': [0.01, 0.05, 0.1, 0.5],
        'subsample': [0.8], 
        'colsample_bytree': [0.8]
    }

    grid_search_cv = GridSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,     
        scoring='balanced_accuracy', 
        cv=loo,
        n_jobs=-1
    )
    grid_search_cv.fit(X_train, y_train)
   
    xgb = grid_search_cv.best_estimator_
    print(grid_search_cv.best_params_)

    X_train_final, X_es_eval, y_train_final, y_es_eval = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
    xgb.set_params(n_estimators=1000, early_stopping_rounds=10)
    
    xgb.fit(X_train_final, y_train_final, eval_set=[(X_es_eval, y_es_eval)], verbose=False)
    y_probs = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(confusion_matrix(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_probs)
    
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
