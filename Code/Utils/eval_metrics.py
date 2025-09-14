# Evaluation Metrics
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

# Number of Errors
def NumberOfErrors(y_actual, y_pred): # Two arrays: actual label and prediction
    return np.sum(y_actual != y_pred)

# accuracy
def Accuracy(y_actual, y_pred): # Two arrays: actual label and prediction
    return (np.sum(y_actual == y_pred)/y_actual.shape[0])*100

# auroc
def AUROC(y_actual, pred_logits):
    # if np.sum(np.isnan(pred_logits)) == 0:
    #     return roc_auc_score(y_actual, pred_logits)*100
    # else:
    #     return np.nan
    # Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes
        exp_logits = np.exp(pred_logits - np.max(pred_logits, axis=1, keepdims=True))
        Y_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return roc_auc_score(y_actual.reshape(-1), Y_probs, average="macro", multi_class='ovr')*100


# Balanced Accuracy
def BalancedAccuracy(y_actual, y_pred):
    return balanced_accuracy_score(y_actual, y_pred)*100

# AUPRC
def AUPRC(y_actual, pred_logits):
    # if np.sum(np.isnan(pred_logits)) == 0:
    #     precision_val, recall_val, _ = precision_recall_curve(y_actual, pred_logits)
    #     return auc(recall_val, precision_val)*100
    # else:
    #     return np.nan
    Y_true_bin = label_binarize(y_actual, classes=np.unique(y_actual))
    return average_precision_score(Y_true_bin, pred_logits, average='macro')*100

def get_all_metrics(Y_true, Y_pred, Y_logits, time_taken):
    num_errors = NumberOfErrors(Y_true, Y_pred)
    accuracy = Accuracy(Y_true, Y_pred)
    auroc = AUROC(Y_true, Y_logits)
    auprc = AUPRC(Y_true, Y_logits)
    balanced_accuracy = BalancedAccuracy(Y_true, Y_pred)

    return {"Num. Errors"   : num_errors,
            "Accuracy"      : accuracy,
            "AUROC"         : auroc,
            "AUPRC"         : auprc,
            "Bal. Accuracy" : balanced_accuracy,
            "Time"          : time_taken}
