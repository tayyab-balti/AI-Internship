## Confusion Matrix
A Confusion Matrix is an N×N table used to evaluate the performance of a classification model, where N is the number of target classes. It compares actual target values with predicted values, providing insights into the model's accuracy.

### Structure
For a binary classification problem, the confusion matrix is a 2×2 table:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

### Key Components
- **TP (True Positive)**: Actual positive and predicted as positive.
- **TN (True Negative)**: Actual negative and predicted as negative.
- **FP (False Positive)**: Actual negative but predicted as positive `(Type I Error)`.
- **FN (False Negative)**: Actual positive but predicted as negative `(Type II Error)`.

### Performance Metrics
- **Accuracy:** Number of predictions the model got right.
  `Formula: (TP + TN) / (TP + TN + FP + FN)`

- **Error Rate:** Number of predictions the model got wrong.
  `Formula: (FP + FN) / (TP + TN + FP + FN)` or `1 - accuracy`

- **Precision:** When the model predicts positives, how often is it right?
  `Formula: (TP / (TP + FP))`

- **Recall:** When it's actually yes (+ve), how often does the model predict yes?
  `Formula: (TP / (TP + FN))`
- **F1 score:** calculates the harmonic mean of precision and recall, providing a balanced measure of a model's accuracy that considers both false positives and false negatives.
  `Formula: 2 * (Precision * Recall) / (Precision + Recall)`

### Applications
- **Medical Diagnosis**: Evaluating the performance of diagnostic tests.
- **Fraud Detection**: Assessing the accuracy of fraud detection systems.

`Real-life scenario:` A hospital uses an AI model to predict disease presence based on patient symptoms. The Confusion Matrix shows correct diagnoses, false alarms, and missed cases, helping doctors assess the model's reliability for clinical use.

### Multi-Class Confusion Matrix

| Actual \ Predicted | A | B | C |
|--------------------|---|---|---|
| A                  | 5 | 1 | 0 |
| B                  | 2 | 3 | 1 |
| C                  | 0 | 1 | 4 |

### Binary Classification Conversion

**Positive (A) & Negative (B & C)**
Converting class A to positive, B and C to negative:

| Actual \ Predicted | Positive (A) | Negative (B & C) |
|--------------------|--------------|------------------|
| Positive (A)       | 5 (TP)       | 1 (FN)           |
| Negative (B & C)   | 2 (FP)       | 9 (TN)           |

**Positive (B) & Negative (A & C)**

| Actual \ Predicted | Positive (B) | Negative (A & C) |
|--------------------|--------------|------------------|
| Positive (B)       | 3 (TP)       | 3 (FN)           |
| Negative (A & C)   | 2 (FP)       | 9 (TN)           |

**Positive (C) & Negative (A & B)**

| Actual \ Predicted | Positive (C) | Negative (A & B) |
|--------------------|--------------|------------------|
| Positive (C)       | 4 (TP)       | 1 (FN)           |
| Negative (A & B)   | 1 (FP)       | 11 (TN)          |
