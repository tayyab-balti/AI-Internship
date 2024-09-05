## Linear Regression

It is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables.

`Real-life scenario`: Predicting house prices based on square footage.

### Multiple Linear Regression
Involves multiple independent variables (X₁, X₂, ... Xₙ) and one dependent variable (Y).

`Equation`: Y = m₁X₁ + m₂X₂ + ... + mₙXₙ + c

Where:
- **Y**: Dependent variable
- **X**: Independent variable
- **m**: Slope/Coefficient (shows how much `y` changes with one-unit change in `X`)
- **c**: Y-intercept (value of `y` when `X` is 0)

`Real-life scenario`: Predicting a car's fuel efficiency based on its weight, horsepower, and age.

### Code
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {'Weight': [1500, 1600, 1700, 1800, 1900],
        'Horsepower': [130, 140, 150, 160, 170],
        'Age': [5, 4, 3, 2, 1],
        'FuelEfficiency': [30, 28, 27, 25, 24]}
df = pd.DataFrame(data)

# Independent (X) and Dependent (Y) variables
X = df[['Weight', 'Horsepower', 'Age']]
y = df['FuelEfficiency']

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Coefficients and Intercept
print(f"Coefficients: {model.coef_}, Intercept: {model.intercept_}")

# Predicting fuel efficiency for a car with 1600kg, 145HP, and 2 years old
predicted_efficiency = model.predict([[1600, 145, 2]])
print(f"Predicted Fuel Efficiency: {predicted_efficiency[0]}")
```


## Polynomial Regression
Polynomial Regression is a powerful technique for modeling non-linear relationships as an nth degree polynomial. While more complex than linear regression, it offers greater flexibility in capturing real-world phenomena that often exhibit curvilinear patterns.

`Equation`: Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε

Where:
- Y is the dependent variable
- X is the independent variable
- β₀, β₁, β₂, ..., βₙ are the coefficients
- n is the degree of the polynomial
- ε is the error term

`Real-life scenario`: Modeling the growth of a plant over time, where it grows slowly at first, accelerates, and then slows down, capturing the non-linear growth pattern.

### Where to Use
- The relationship between variables is known to be non-linear
- Linear regression doesn't provide a good fit to the data
- You need to model curved relationships
- The data shows clear non-linear trends when plotted


### Code
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# Transform the features to polynomial (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict using the polynomial model
y_pred = model.predict(X_poly)

# Plot the results
plt.scatter(X, y, color='blue')  # Original data points
plt.plot(X, y_pred, color='red')  # Fitted polynomial curve
plt.title('Polynomial Regression (degree 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```


## Logistic Regression
Logistic regression is used for binary classification where we use sigmoid function, that takes input as independent variables and produces a probability value between 0 and 1.

`Equation`: P(Y=1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))

Where:
- P(Y=1) is the probability of the dependent variable being 1
- e is the base of natural logarithms
- β₀, β₁, ..., βₙ are the regression coefficients
- X₁, X₂, ..., Xₙ are the independent variables

### Key Characteristics
- Used for classification problems
- Outcome is binary (0 or 1, Yes or No, True or False)
- Produces a S-shaped curve (logistic curve) instead of a straight line

`Real-life scenario`: A bank uses logistic regression to predict the likelihood of a customer defaulting on payments based on attributes like credit score, income, and credit history. This helps automate the credit card approval process efficiently.

### Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample Data
data = {
    'Credit_Score': [650, 700, 620, 720, 630, 710, 680, 640, 750, 690],
    'Annual_Income': [40000, 60000, 35000, 80000, 45000, 75000, 55000, 38000, 90000, 50000],
    'Years_Credit_History': [3, 8, 2, 10, 4, 9, 7, 2, 12, 6],
    'Approval': [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # 1 = Approved, 0 = Rejected
}

df = pd.DataFrame(data)

# Independent variables (features) and dependent variable (target)
X = df[['Credit_Score', 'Annual_Income', 'Years_Credit_History']]
y = df['Approval']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predicting approval for a new customer
new_customer = [[690, 50000, 5]]  # Credit score, Annual Income, Years of Credit History
prediction = model.predict(new_customer)
print(f"Credit card approval prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
```


