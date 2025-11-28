import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import confusion_matrix, roc_auc_score

def save_confusion_matrix(y_true, y_pred, labels, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

def compute_auc(y_true, y_scores):
    try:
        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        return None
