# Introduction

Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables.

`Real-life scenario`: Predicting house prices based on square footage.

## Simple Linear Regression
Involves one independent variable (X) and one dependent variable (Y).

`Formula`: Y = mX + c

- **Y**: Dependent variable
- **X**: Independent variable
- **m**: Slope (shows how much `y` changes with one-unit change in `X`)
- **c**: Y-intercept (value of `y` when `X` is 0)

`Real-life scenario`: Estimating a person's salary (Y) based on their experience (X).

## Multiple Linear Regression
Involves multiple independent variables (X1, X2, ..., Xn) and one dependent variable (Y).

`Formula`: Y = m1X1 + m2X2 + ... + bnXn + c

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
