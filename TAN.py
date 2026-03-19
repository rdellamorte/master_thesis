import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import TreeSearch, BayesianEstimator
from sklearn.model_selection import train_test_split, LeaveOneOut 
from sklearn.metrics import classification_report, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
import warnings
import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)

def loocv_fold(train_idx, val_idx, X_train, y_train, dag_edges, prior_type, target_name, num_parameter):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    
    df_train_fold = X_train_fold.copy()
    df_train_fold[target_name] = y_train_fold
    
    tan_fold = DiscreteBayesianNetwork(dag_edges)
    
    if prior_type == 'dirichlet':
        tan_fold.fit(df_train_fold, estimator=BayesianEstimator, prior_type=prior_type, pseudo_counts=num_parameter)
    else:
        tan_fold.fit(df_train_fold, estimator=BayesianEstimator, prior_type=prior_type, equivalent_sample_size=num_parameter)
    
    prob = tan_fold.predict_probability(X_val_fold).iloc[0, 1]
    return prob, y_train.iloc[val_idx[0]]

def loocv_iteration(X_train, y_train, prior_type, target_name='surv_1yr', num_parameter = 1):
    df_full_train = X_train.copy()
    df_full_train[target_name] = y_train
    est = TreeSearch(df_full_train)
    dag = est.estimate(estimator_type='tan', class_node=target_name)
    dag_edges = list(dag.edges())
    loo = LeaveOneOut()

    results = Parallel(n_jobs=-1)(delayed(loocv_fold)(
        train_idx, val_idx, X_train, y_train, dag_edges, prior_type, target_name, num_parameter
    ) for train_idx, val_idx in loo.split(X_train))
    
    all_probs, all_true = zip(*results)  

    all_true_np = np.array(all_true)
    all_probs_np = np.array(all_probs)
    all_preds_np = (all_probs_np > 0.6).astype(float)

    auc_combo = roc_auc_score(all_true_np, all_probs_np)
    f1_1_combo = f1_score(all_true_np, all_preds_np)
    f1_0_combo = f1_score(all_true_np, all_preds_np, pos_label=0)
    accuracy_combo = balanced_accuracy_score(all_true_np, all_preds_np)
    
    pred_pos_combo = (all_preds_np == 1).sum()
    pred_neg_combo = (all_preds_np == 0).sum()
    param_metrics_dict = {     
        'prior_type': prior_type,
        'pseudo_count': pseudo_count_val,   
        'equivalent_sample_size': sample_size_val,                     
        'auc': auc_combo,
        'accuracy': accuracy_combo,
        'f1_1': f1_1_combo,
        'f1_0': f1_0_combo,
        'pred_pos': pred_pos_combo,
        'pred_neg': pred_neg_combo
        }
    
    return param_metrics_dict

tqdm.__init__ = partial(tqdm.__init__, disable=True)
base_dir = Path(__file__).resolve().parent

df = pd.read_csv(f'{base_dir.parent}\\dataset\\dataset_integrato.csv', sep=',')
target = 'surv_1yr'
df[target] = df[target].astype('category')

X = df.drop(columns=['event_death','ID','surv_1yr','surv_2yr','surv_3yr','surv_4yr','surv_5yr'])

age_limits = [0, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf] 
age_labels = ['under40', '40to45', '45to50', '50to55', '55to60', '60to65', '65to70', '70to75', '75to80', '80to85', 'over85']
X['age'] = pd.cut(
    X['age'], 
    bins=age_limits, 
    labels=age_labels, 
    right=False, 
    include_lowest=True 
)

cols = X.columns
X[cols] = X[cols].astype('category')

y = df[target]

auc_scores = []
f1_1 = []
f1_0 = []
f1_com = []
prior_types = ['dirichlet', 'BDeu'] 
pseudo_counts = [0.1, 1, 1.5, 3] 
eq_sample_sizes = [0.1, 1, 2]

for iteration in range(1, 11):
    print(f"Starting iteration {iteration}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    param_metrics_acc = []
    loo = LeaveOneOut()

    df_train = X_train.copy()
    df_train[target] = y_train

    for prior_type in prior_types:

        if (prior_type == 'dirichlet'):
            for pseudo_count_val in pseudo_counts:        
                sample_size_val = 0   
                param_metrics_dict = loocv_iteration(X_train, y_train, prior_type, target, pseudo_count_val)
                param_metrics_acc.append(param_metrics_dict)
            
        elif (prior_type == 'BDeu'):
            pseudo_count_val = 1
            for sample_size_val in eq_sample_sizes:
                param_metrics_dict = loocv_iteration(X_train, y_train, prior_type, target, sample_size_val)
                param_metrics_acc.append(param_metrics_dict)

    param_metrics_df = pd.DataFrame(param_metrics_acc)

    numeric_cols = param_metrics_df.select_dtypes(include=['number']).columns
    param_metrics_df[numeric_cols] = param_metrics_df[numeric_cols].round(2)

    best_config = param_metrics_df.sort_values(by='accuracy', ascending=False).iloc[0]
    best_prior = best_config['prior_type']
    best_pseudo_count = best_config['pseudo_count']
    best_sample_size = best_config['equivalent_sample_size']
    print(f"Best configuration: prior {best_prior}, pseudo count {best_pseudo_count}, sample size {best_sample_size}") 

    df_train = X_train.copy()
    df_train[target] = y_train

    est = TreeSearch(df_train)
    dag = est.estimate(estimator_type='tan', class_node=target)
    tan = DiscreteBayesianNetwork(dag.edges()) 

    if (best_prior == 'dirichlet'):
        tan.fit(df_train, estimator=BayesianEstimator, prior_type=best_prior, pseudo_counts=best_pseudo_count) 
    elif (best_prior == 'BDeu'):
        tan.fit(df_train, estimator=BayesianEstimator, prior_type=best_prior, equivalent_sample_size=best_sample_size)

    y_prob = tan.predict_probability(data=X_test).iloc[:, 1].values 
    y_pred = (y_prob >= 0.75).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(confusion_matrix(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_prob)
    auc_df = pd.DataFrame(['auc', auc_score]) 

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
