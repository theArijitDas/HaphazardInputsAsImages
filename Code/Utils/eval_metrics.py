import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

def NumberOfErrors(Y_true, Y_pred):
    res = (Y_true.reshape(-1) != Y_pred.reshape(-1))
    return int(sum(res))

def Accuracy(Y_true, Y_pred):
    return accuracy_score(Y_true, Y_pred)

def AUROC(Y_true, Y_logits):
    nan_index = np.where(np.isnan(Y_logits))[0]
    if nan_index.size:
        return np.nan

    # Binary case
    if np.unique(Y_true).shape[0] == 2:
        return roc_auc_score(Y_true, Y_logits)

    # Multiclass case
    y_probs = softmax(Y_logits, axis=1)  # shape (N,C)
    classes = np.unique(Y_true)
    Y_true_bin = label_binarize(Y_true, classes=classes)
    return roc_auc_score(Y_true_bin, y_probs, multi_class="ovr")

def AUPRC(Y_true, Y_logits):
    nan_index = np.where(np.isnan(Y_logits))[0]
    if nan_index.size:
        return np.nan
    
    # Binary case
    if np.unique(Y_true).shape[0] == 2:
        return average_precision_score(Y_true, Y_logits)

    # Multiclass case
    Y_probs = softmax(Y_logits, axis=1)
    classes = np.unique(Y_true)
    y_true_bin = label_binarize(Y_true, classes=classes)
    return average_precision_score(y_true_bin, Y_probs, average="macro")

def BalancedAccuracy(Y_true, Y_pred):
    return balanced_accuracy_score(Y_true, Y_pred)

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
