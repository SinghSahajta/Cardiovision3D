# Evaluation results (out-of-fold, nested stratified 5-fold CV)

Threshold-independent ranking quality:

| Target | Prevalence | ROC-AUC | PR-AUC | Brier |
| :--- | :---: | :---: | :---: | :---: |
| CAD | 71.3% | 0.9070 | 0.9558 | 0.1067 |
| LAD | 58.4% | 0.8374 | 0.8729 | 0.1601 |
| LCX | 39.3% | 0.7415 | 0.6156 | 0.2049 |
| RCA | 37.6% | 0.6991 | 0.5370 | 0.2233 |

Operating point - default threshold 0.5:

| Target | Threshold | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CAD | 0.500 | 0.8548 | 0.8739 | 0.9306 | 0.9013 |
| LAD | 0.500 | 0.7789 | 0.7957 | 0.8362 | 0.8154 |
| LCX | 0.500 | 0.6964 | 0.6080 | 0.6387 | 0.6230 |
| RCA | 0.500 | 0.6634 | 0.5652 | 0.4561 | 0.5049 |

Operating point - tuned threshold:

| Target | Threshold | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CAD | 0.659 | 0.8713 | 0.9155 | 0.9028 | 0.9091 |
| LAD | 0.529 | 0.7855 | 0.8111 | 0.8249 | 0.8179 |
| LCX | 0.353 | 0.6436 | 0.5307 | 0.7983 | 0.6376 |
| RCA | 0.229 | 0.6073 | 0.4855 | 0.7368 | 0.5854 |
