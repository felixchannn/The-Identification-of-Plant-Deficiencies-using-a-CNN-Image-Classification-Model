'''

what the evaluation matrix needs

rows = true
columns = predicted

1. import libraries
2. define my classes
3. project-specific cost matrix
4. utility functions
5. argument parsing (CLI)
6. output directory setup
7. data transforms and loaders
8. resolve final class list
9. device selection
10. model definition and weight loading
11. prediction loop
12. metrics computation
13. visualisations
14. export per class metrics 
15. text report
16. console summary
17. entrypoint


to do
- what metrics are necessary to valid my model?
- what are the metrics i will use?
- metrics to access
- internal metrics e.g. confusion matrix (above)
- 

- need to research some other metrics 


Per-class Recall (Sensitivity)
- Of all actual leaves of class k, how many you correctly catch. This is the key “don’t 
miss sick plants” metric. Aim high (e.g., ≥0.90) for each deficiency you care about.

Balanced Accuracy
- Mean of recall across classes. Handles class imbalance and tells you how fairly the 
model performs across all deficiencies (not just the common ones).

Macro F1
- Harmonic mean of precision & recall per class, then averaged equally across classes. 
Rewards models that both find and are right about rare deficiencies.

Macro AUPRC (Area Under Precision–Recall Curve)
- Threshold-free measure that’s more informative than AUROC when some deficiencies are rare. 
Reflects performance across all possible decision thresholds.

Calibration (ECE: Expected Calibration Error)
- Checks whether a “0.85 confidence” prediction is correct ~85% of the time. Crucial if you’ll 
set action thresholds or show confidence to agronomists.

'''