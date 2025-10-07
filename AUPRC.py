from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# y_true: (N,) 0/1 for a given class (e.g., nitrogen vs rest)
# y_proba: (N,) model probability for that class
precision, recall, thresh = precision_recall_curve(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)  # ≈ AUPRC summary

print("Average Precision (≈ AUPRC):", round(ap, 3))
# For multiclass: compute AP per class and take the macro average.
