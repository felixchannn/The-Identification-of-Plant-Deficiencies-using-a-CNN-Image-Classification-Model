from sklearn.metrics import matthews_corrcoef

# y_true, y_pred: lists/arrays of ints or strings (same label space)
mcc = matthews_corrcoef(y_true, y_pred)
print("MCC:", round(mcc, 4))
