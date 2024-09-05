# Introduction

Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables.

`Real-life scenario`: Predicting house prices based on square footage.

## Multiple Linear Regression
Involves multiple independent variables (X₁, X₂, ... Xₙ) and one dependent variable (Y).

`Equation`: Y = m₁X₁ + m₂X₂ + ... + mₙXₙ + c

Where:
- **Y**: Dependent variable
- **X**: Independent variable
- **m**: Slope/Coefficient (shows how much `y` changes with one-unit change in `X`)
- **c**: Y-intercept (value of `y` when `X` is 0)

`Real-life scenario`: Predicting a car's fuel efficiency based on its weight, horsepower, and age.

```python
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

## Where to Use
1. The relationship between variables is known to be non-linear
2. Linear regression doesn't provide a good fit to the data
3. You need to model curved relationships
4. The data shows clear non-linear trends when plotted


## Code
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
