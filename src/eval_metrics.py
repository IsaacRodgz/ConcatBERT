import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def metrics(results, truths):
    preds = results.cpu().detach().numpy()
    truth = truths.cpu().detach().numpy()
    
    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    f_score = f1_score(truth, preds, average='micro')
    accuarcy = accuracy_score(truth, preds)

    return accuarcy, f_score

def multiclass_acc(results, truths):
    preds = results.view(-1).cpu().detach().numpy()
    truth = truths.view(-1).cpu().detach().numpy()
    
    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    return np.sum(preds == truths) / float(len(truths))