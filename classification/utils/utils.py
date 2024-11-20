from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve, top_k_accuracy_score
import matplotlib.pyplot as plt
import os
import fnmatch


def get_metrics(thresholds, y_true, y_scores):
    y_pred = []
    for threshold in thresholds:
        predictions = (y_scores >= threshold)
        y_pred.append(predictions)
    f1s = []
    accuracies = []
    precisions = []
    recalls = []
    tns = []
    fps = []
    fns = []
    tps = []
    tprs = []
    fprs = []
    for i in range(len(y_pred)):
        f1 = f1_score(y_true, y_pred[i])
        accuracy = accuracy_score(y_true, y_pred[i])
        precision = precision_score(y_true, y_pred[i], zero_division=0)
        recall = recall_score(y_true, y_pred[i], zero_division=0)
        con_m = confusion_matrix(y_true, y_pred[i])
        tn, fp, fn, tp = con_m.ravel()
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        f1s.append(f1)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        tprs.append(tpr)
        fprs.append(fpr)
    return f1s, accuracies, precisions, recalls, tns, fps, fns, tps, tprs, fprs


def plot_roc_curve(epoch, y_true, y_scores, desc, is_save=True):
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC')
    plt.legend(loc="lower right")
    if is_save:
        path = f'./metrics'
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(f'{path}/{desc}_epoch{epoch}_roc{roc_auc:.3f}.pdf', format='pdf')
    else:
        plt.show()
    return roc_auc


def get_resume_model_path(epoch):
    for root, _, files in os.walk('./checkpoints'):
        for file in files:
            if fnmatch.fnmatch(file, f'*epoch{epoch}*'):
                return os.path.join(root, file)