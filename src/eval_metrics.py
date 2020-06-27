import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def eval_hateful_meme(results, truths):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='micro')
    accuarcy = accuracy_score(results, truths)

    print("F1 score: ", f_score)
    print("Accuracy: ", accuarcy)

    print("-" * 50)