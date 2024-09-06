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

### Applications
- Text classification (spam detection, sentiment analysis)
- Face recognition, Weather forecasting, News categorization

`Real-life scenario:` Predicting spam emails based on features like keywords (free, discounts), sender address, and message length. It calculates the probability of an email being spam based on these features.

### Code
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Sample data
X = [[10, 20], [15, 30], [25, 45], [35, 55]]  # Features
y = [0, 0, 1, 1]  # Labels (0 = not spam, 1 = spam)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)
```


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

### Code
```python
from sklearn import svm
from sklearn.model_selection import train_test_split

# Sample data
X = [[10, 20], [15, 30], [25, 45], [35, 55]]
y = [0, 0, 1, 1]  # 0 = not spam, 1 = spam

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)
```
