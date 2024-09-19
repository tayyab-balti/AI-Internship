## Naive Bayes Classification
Naive Bayes is a supervised learning algorithm used for solving classification problems. It is a probabilistic classifier that predicts based on the probability of an object belonging to a particular class.

### Key Concepts
- **Naive**: Assumes that all attributes (features) are independent of each other (which may not always be true in practice).
- **Bayes**: Based on `Bayes' Theorem`, which calculates the conditional probability of an event.

### Bayes' Theorem
P(A|B) = P(B|A) * P(A) / P(B)

Where:
- A, B are Events
- P(A|B) is the posterior probability
- P(B|A) is the likelihood
- P(A) is the prior probability
- P(B) is the marginal likelihood

### Steps:
1) Prior Probability:
    - P(fever=yes) = 7/10
    - P(fever=no) = 3/10

2) Conditional Probability:
```
|       | Yes  | No  |
|--------------|-----|
| Covid | 4/7  | 2/3 |
| Flu   | 3/7  | 2/3 |
```

`Example:` Here yes & no refers to Fever:
- P(yes|flu,Covid) = P(flu/yes) * P(covid/yes) * P(yes)
- P(No|flu,Covid) = P(flu/No) * P(covid/No) * P(No)

### Applications
- Text classification (spam detection, sentiment analysis)
- Face recognition, Weather forecasting, News categorization

`Real-life scenario:` Predicting spam emails based on features like keywords (free, discounts), sender address, and message length. It calculates the probability of an email being spam based on these features.

### Visualization
![Naive Bayes Flowchart](https://raw.githubusercontent.com/tayyab-balti/AI-Internship/master/Images/naive-bayes-flowchart-svg.svg)


## Support Vector Machines (SVM)
Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks. It creates a `hyperplane` to separate data points into different classes, with margins to maximize separation.

### Key Concepts
- **Hyperplane**: A decision boundary that separates the data into classes. In 2D, it's a line; in 3D, it's a plane; in higher dimensions, it's a hyperplane.
- **Support Vectors**: Data points closest to the hyperplane that influence its position and orientation.
- **Margin**: The distance between the hyperplane and the nearest data points from either class.

### How it Works
1. SVM creates two marginal lines parallel to the hyperplane, passing through the nearest data samples of each class.
2. The algorithm seeks to maximize the distance between these marginal lines.
3. The goal is to maximize the margin between the classes.

### Types of SVM
- **Linear SVM**: Used when data can be separated into two classes with a straight line.
- **Non-linear SVM**: Used when data is not linearly separable. Employs kernel functions to transform the data into a higher-dimensional space.

### Kernel Functions
- Transform low-dimensional input space into a higher-dimensional space.
- Convert non-separable problems into separable ones.
- `Common kernel functions:` Polynomial, Sigmoid kernel, Radial Basis Function (RBF)

### Applications
- Text and image classification, Face detection, Financial analysis

`Real-life scenario:` Classifying whether an email is **spam or not** based on features such as frequency of words and links.

### Visualization
![Naive Bayes Flowchart](https://raw.githubusercontent.com/tayyab-balti/AI-Internship/master/Images/svm-email-classification.svg)


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
- **Accuracy**: Number of predictions the model got right.
  `Formula: (TP + TN) / (TP + TN + FP + FN)`

- **Error Rate**: Number of predictions the model got wrong.
  `Formula: (FP + FN) / (TP + TN + FP + FN)` or `1 - accuracy`

- **Precision**: When the model predicts positives, how often is it right?
  `Formula: (TP / (TP + FP))`

- **Recall**: When it's actually yes (+ve), how often does the model predict yes?
  `Formula: (TP / (TP + FN))`

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


## Overfitting and Underfitting

Model performance is evaluated based on two factors:
- **Accuracy**: Measures how well a model predicts the correct output.
- **Generalization**: Assesses how well a model performs on new, unseen data.

### Underfitting
Underfitting occurs when a model has low accuracy on both training and test data.

### Causes
- Insufficient data
- Overly simplistic model (e.g., linear model for nonlinear data)
- Inadequate feature selection

### Underfitting Solutions
- Increase model complexity
- Feature engineering
- Gather more relevant data

### Overfitting
Overfitting happens when a model has high accuracy on training data but low accuracy on test data.

### Causes
- Overly complex model
- Training on noise or outliers
- Insufficient regularization

### Right Fit (Good Fit)
A model is considered to have a good fit when it makes predictions with minimal error and generalizes well to new data.

### Characteristics
- Balance between bias and variance
- Good performance on training data
- Good generalization to test data

### Overfitting Solutions
- Regularization techniques (L1, L2)
- Cross-validation
- Ensemble methods
- Early stopping in iterative algorithms

`Real-life scenario:` A weather prediction model undergoes fitting analysis. An underfit model might always predict "sunny," while an overfit model remembers specific dates instead of weather patterns. A well-fit model balances historical patterns with adaptability to new conditions.
