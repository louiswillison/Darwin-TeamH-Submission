import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def save_roc_curve_2(y_test, y_pred_proba, name=None):
    timestamp = str(time.time()).split(".")[0]
    if name is not None:
        file_name = name
    else:
        file_name = "roc_" + timestamp

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    try:
        if ".png" in file_name:
            file_name = file_name.split(".")[0]
        plt.savefig("results/" + file_name + ".png")
        plt.close()

        # print(f"ROC curve saved to results/" + file_name + ".png")
    except Exception as e:
        print("Error saving ROC curve")
        print(e)
        # pass
        # if args.verbose:
        #     print(e)


def save_confusion_matrix(y_test, y_pred, name=None):
    cm = confusion_matrix(y_test, y_pred)

    timestamp = str(time.time()).split(".")[0]
    if name is not None:
        file_name = name
    else:
        file_name = "cm_" + timestamp

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.grid(False)

    try:
        if ".png" in file_name:
            file_name = file_name.split(".")[0]
        plt.savefig("results/" + file_name + ".png")
        plt.close()

        # print(f"Confusion matrix saved to results/" + file_name + ".png")

    except Exception as e:
        print("Error saving confusion matrix")
        print(e)
        # if args.verbose:
        #     print(e)]
