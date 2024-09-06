## Naive Bayes Classification
Naive Bayes is a supervised learning algorithm used for solving classification problems. It is a probabilistic classifier that predicts based on the probability of an object belonging to a particular class.

### Key Concepts
- **Naive**: Assumes that all attributes (features) are independent of each other (which may not always be true in practice).
- **Bayes**: Based on `Bayes' Theorem`, which calculates the conditional probability of an event.

### Bayes' Theorem
P(A|B) = P(B|A) * P(A) / P(B)

Where:
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
