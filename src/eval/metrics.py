import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

def evaluate_probs(y_true, proba_3):
    y_pred = proba_3.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    ll  = log_loss(y_true, proba_3, labels=[0,1,2])
    # multiclass Brier (mean squared error vs one-hot)
    Y = np.eye(3)[y_true]
    brier = np.mean((proba_3 - Y)**2)
    return {"accuracy":float(acc), "macro_f1":float(f1), "log_loss":float(ll), "brier":float(brier)}
 
