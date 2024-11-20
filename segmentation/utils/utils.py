from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve, top_k_accuracy_score
import matplotlib.pyplot as plt
import os
import fnmatch
from losses.criterion import dice_score
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import random


def get_metrics(thresholds, y_true, y_scores):
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
    dices = []
    for threshold in thresholds:
        y_pred = (y_scores > threshold)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        con_m = confusion_matrix(y_true, y_pred)
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
    return dices, f1s, accuracies, precisions, recalls, tns, fps, fns, tps, tprs, fprs


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


def get_dice(thresholds, y_scores, y_trues, mode='mean'):
    dices = []
    max_dice = 0.0
    max_threshold = 0.0
    plt.rcParams['figure.max_open_warning'] = 100
    if mode == 'mean':
        for threshold in thresholds:
            y_pred = (y_scores > threshold)
            dice = dice_score(y_pred, y_trues)
            dices.append(dice)
            if dice > max_dice:
                max_dice = dice
                max_threshold = threshold
    elif mode == 'every':
        for threshold in thresholds:
            dice = []
            for i in range(y_trues.shape[0]):
                score = y_scores[i]
                true = y_trues[i]
                pred = (score > threshold)
                d = dice_score(pred, true)
                dice.append(d)
            dices.append(dice)
        mean_dice = np.mean(np.array(dices), axis=1)
        max_index = np.argmax(mean_dice)
        max_dice = mean_dice[max_index]
        max_threshold = thresholds[max_index]
    return dices, max_dice, max_threshold


def gethausdorff_distance(threshold, y_scores, y_trues):
    haus_dists = []
    indexes = list(range(y_trues.shape[0]))

    # 打乱列表顺序
    random.shuffle(indexes)
    # indexes = np.random.choice(np.arange(y_trues.shape[0]), size=4, replace=False)
    i = 0
    for index in indexes:
        pred_binary = y_scores[index]
        pred_binary = pred_binary.squeeze(0)

        points_label = y_trues[index]
        points_label = points_label.squeeze(0)

        points_pred = np.argwhere(pred_binary >= threshold)
        points_label = np.argwhere(points_label == 1)
        if len(points_pred) == 0 or len(points_label) == 0 or len(points_pred) > 500000 or len(points_label) > 500000:
            continue
        i += 1

        label_to_pred = directed_hausdorff(points_label, points_pred)[0]
        pred_to_label = directed_hausdorff(points_pred, points_label)[0]
        hausdorff_dist = np.max([label_to_pred, pred_to_label])
        haus_dists.append(hausdorff_dist)
        # print(f'hausdorff_dist={hausdorff_dist}')
    return np.mean(haus_dists) if len(haus_dists) > 0 else 0

