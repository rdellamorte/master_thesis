import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score, balanced_accuracy_score
from tqdm import tqdm
from functools import partial
from pathlib import Path

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )
        torch.nn.init.xavier_uniform_(self.network[0].weight)
        torch.nn.init.xavier_uniform_(self.network[2].weight)
        torch.nn.init.zeros_(self.network[0].bias)
        torch.nn.init.zeros_(self.network[2].bias)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits   

tqdm.__init__ = partial(tqdm.__init__, disable=True)
base_dir = Path(__file__).resolve().parent

df = pd.read_csv(f'{base_dir.parent}\\dataset\\dataset_integrato.csv', sep=',')

X = df.drop(columns=['event_death','ID','surv_1yr','surv_2yr','surv_3yr','surv_4yr','surv_5yr'])

X = pd.get_dummies(X, columns=['tnm8', 'region', 'smoking', 'keck', 'dececco','chemo', 'surg', 'radio', 'sex'])
X = X.astype(float)

y = df['surv_1yr']
pos_weight = torch.tensor([((y == 0).sum() / (y == 1).sum())])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

h_sizes = [2, 3, 5]                 
lr_vals = [0.1, 0.05, 0.01]          
wd_vals = [0.01, 0.001, 0.0001]     #higher values make the model unstable

auc_scores = []
f1_1 = []
f1_0 = []
f1_com = []

for iteration in range(1, 11):
    print(f"Starting iteration {iteration}")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    X_train_t = torch.FloatTensor(X_train.values)
    X_test_t = torch.FloatTensor(X_test.values)
    y_train_t = torch.FloatTensor(y_train.values.copy()).view(-1, 1)
    y_test_t = torch.FloatTensor(y_test.values.copy()).view(-1, 1)

    param_metrics_acc = []
    loo = LeaveOneOut()

    for h_size in h_sizes:
        for lr_val in lr_vals:
            for wd_val in wd_vals:
                
                all_probs = []
                all_true = []

                for train_idx, val_idx in loo.split(X_train_t):
                    X_train_fold = X_train_t[train_idx]
                    y_train_fold = y_train_t[train_idx]
                    X_val_fold = X_train_t[val_idx]
                    y_val_fold = y_train_t[val_idx]

                    scaler = StandardScaler()
                    X_train_fold = torch.FloatTensor(scaler.fit_transform(X_train_fold))
                    X_val_fold = torch.FloatTensor(scaler.transform(X_val_fold))

                    model = NeuralNetwork(input_size = X.shape[1], hidden_size=h_size)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd_val)

                    param_metrics_dict = {}
                    model.train()
                    for epoch in range(150):
                        scores = model(X_train_fold)
                        loss = criterion(scores, y_train_fold)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval() 

                    with torch.no_grad():        
                        output_scores = model(X_val_fold)
                        probs = torch.sigmoid(output_scores)
                        predictions = (torch.sigmoid(output_scores) > 0.6).float()
                        all_probs.append(probs.item())
                        all_true.append(y_val_fold.item())                    

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
                            'h_size': h_size,
                            'lr': lr_val,
                            'wd': wd_val,
                            'auc': auc_combo,
                            'accuracy': accuracy_combo,
                            'f1_1': f1_1_combo,
                            'f1_0': f1_0_combo,
                            'pred_pos': pred_pos_combo,
                            'pred_neg': pred_neg_combo
                            }
                        
                param_metrics_acc.append(param_metrics_dict)

    param_metrics_df = pd.DataFrame(param_metrics_acc)

    numeric_cols = param_metrics_df.select_dtypes(include=['number']).columns
    numeric_cols = numeric_cols.drop('wd')
    param_metrics_df[numeric_cols] = param_metrics_df[numeric_cols].round(2)

    scaler = StandardScaler()
    X_train_t = torch.FloatTensor(scaler.fit_transform(X_train_t)) 
    X_test_t = torch.FloatTensor(scaler.transform(X_test_t)) 

    best_config = param_metrics_df.sort_values(by='accuracy', ascending=False).iloc[0]
    best_h_size = int(best_config['h_size'])
    best_lr_val = float(best_config['lr'])
    best_wd_val = float(best_config['wd'])

    print(best_config)

    final_model = NeuralNetwork(input_size = X.shape[1], hidden_size=best_h_size)
    final_optimizer_auc = torch.optim.Adam(final_model.parameters(), lr=best_lr_val, weight_decay=best_wd_val)
    final_model.train()

    for epoch in range(150):
        scores = final_model(X_train_t)
        loss = criterion(scores, y_train_t)
        final_optimizer_auc.zero_grad()
        loss.backward()
        final_optimizer_auc.step()

    final_model.eval() 

    with torch.no_grad():        
        output_scores = final_model(X_test_t)
        probs = torch.sigmoid(output_scores)
        predictions = (torch.sigmoid(output_scores) > 0.6).float()

        y_true = y_test_t.numpy()
        y_pred = predictions.numpy()
        auc = roc_auc_score(y_true, probs)     
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
        report = classification_report(y_test, y_pred, output_dict=True)

    f1_1.append(report['1']['f1-score']) 
    f1_0.append(report['0']['f1-score'])
    f1_com.append(report['weighted avg']['f1-score'])
    auc_scores.append(auc)

metrics_dict = {
    'mean': [np.mean(auc_scores), np.mean(f1_0), np.mean(f1_1), np.mean(f1_com)],
    'std_dev': [np.std(auc_scores), np.std(f1_0), np.std(f1_1), np.std(f1_com)]
}
rows = ['auc', 'f1_0', 'f1_1', 'f1_com']
df_metrics = pd.DataFrame(metrics_dict, index=rows)
df_metrics = df_metrics.round(2)
print(df_metrics)
