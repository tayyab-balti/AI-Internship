## Linear Regression

It is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables.

`Real-life scenario:` Predicting house prices based on square footage.

### Multiple Linear Regression
Involves multiple independent variables (X₁, X₂, ... Xₙ) and one dependent variable (Y).

`Equation:` Y = m₁X₁ + m₂X₂ + ... + mₙXₙ + c

Where:
- **Y**: Dependent variable
- **X**: Independent variable
- **m**: Slope/Coefficient (shows how much `y` changes with one-unit change in `X`)
- **c**: Y-intercept (value of `y` when `X` is 0)

### Visualization
![Linear Regression Flowchart](https://raw.githubusercontent.com/tayyab-balti/AI-Internship/master/Images/decision-tree-entropy-visualization.svg)

`Real-life scenario:` Predicting a car's fuel efficiency based on its weight, horsepower, and age.

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

`Equation:` Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε

Where:
- Y is the dependent variable
- X is the independent variable
- β₀, β₁, β₂, ..., βₙ are the coefficients
- n is the degree of the polynomial
- ε is the error term

`Real-life scenario:` Modeling the growth of a plant over time, where it grows slowly at first, accelerates, and then slows down, capturing the non-linear growth pattern.

### Where to Use
- The relationship between variables is known to be non-linear
- Linear regression doesn't provide a good fit to the data
- You need to model curved relationships
- The data shows clear non-linear trends when plotted


## Logistic Regression
Logistic regression is used for binary classification where we use sigmoid function, that takes input as independent variables and produces a probability value between 0 and 1.

`Equation:` P(Y=1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))

Where:
- P(Y=1) is the probability of the dependent variable being 1
- e is the base of natural logarithms
- β₀, β₁, ..., βₙ are the regression coefficients
- X₁, X₂, ..., Xₙ are the independent variables

### Key Characteristics
- Used for classification problems
- Outcome is binary (0 or 1, Yes or No, True or False)
- Produces a S-shaped curve (logistic curve) instead of a straight line

`Real-life scenario:` A bank uses logistic regression to predict the likelihood of a customer defaulting on payments based on attributes like credit score, income, and credit history. This helps automate the credit card approval process efficiently.


## K Nearest Neighbors (KNN)
KNN is a simple, non-parametric algorithm used for classification and regression tasks. It predicts the class or value of a data point by looking at the 'K' nearest data points and using majority voting (for classification) or averaging (for regression).

### Key Steps

1. **Calculate Euclidean Distance**: √((x2 - x1)² + (y2 - y1)²)
2. **Select K Nearest Neighbors**:
   The 'K' closest points based on distance (e.g., K=3).
3. **Majority Voting**:
   For K ≥ 3, the class is decided by majority voting or averaging (for regression).

### Visualization
![KNN Flowchart](https://raw.githubusercontent.com/tayyab-balti/AI-Internship/master/Images/knn-visualization.svg)

### Applications
- **Classification**:
  Predicts categories (e.g., whether a user will like a movie).
- **Regression**:
  Predicts continuous values (e.g., estimating a user’s rating for a movie).

### Key Points
- Works for both classification and regression.
- Sensitive to the choice of K and the scale of the data.


## Decision Trees
A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It splits the data into subsets based on the most significant feature.

### Components
- **Root Node**: Top decision node, represents entire dataset.
- **Internal Nodes**: Decision nodes that splits based on a feature.
- **Leaf Nodes**: Final node that represents the predicted outcome.
- **Branches**: Connections between nodes representing the decision flow.

### Key Concepts:
- **Splitting:** Process of dividing a node into sub-nodes
- **Entropy:** Measure of impurity/disorder in the data
- **Information Gain:** Reduction in entropy after a split
- **Gini Impurity:** Alternative to entropy (for CART algorithm)
- **Pruning:** Process of removing unnecessary branches to prevent overfitting

### Visualization 
![Decision Trees Flowchart](https://raw.githubusercontent.com/tayyab-balti/AI-Internship/master/Images/decision-tree-entropy-visualization.svg)

### Building Process Steps:
1) Calculate entropy (H) of entire dataset

$$
H(S) = S{+9, -5} = - \frac{9}{14} \log_2 \left(\frac{9}{14}\right) - \frac{5}{14} \log_2 \left(\frac{5}{14}\right)
$$
- Here +9 = total yes in the dataset, -5 = total no, 14 = total_dataset_rows

2) Calculate entropy for each attribute

$$
H(Sunny) = - \frac{2}{5} \log_2 \left(\frac{2}{5}\right) - \frac{3}{5} \log_2 \left(\frac{3}{5}\right)
H(Rainy) = - \frac{3}{5} \log_2 \left(\frac{3}{5}\right) - \frac{2}{5} \log_2 \left(\frac{2}{5}\right)
$$

3) Calculate Information Gain for each split

$$
IG(S) = H(S) - \left( \frac{5}{14} H(Sunny) \right) - \left( \frac{5}{14} H(Rainy) \right)
$$

4) Choose attribute with highest Information Gain
- Repeat steps 2-4 for each branch until stopping criteria met

`Real-life scenario:` Predicting whether a customer will `churn` based on attributes like usage, customer service calls, and contract duration.


## Random Forest
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class (for classification) or mean prediction (for regression) of the individual trees, and merges their results for more accurate and stable predictions.

### How It Works
1. Create bootstrap dataset from original data by randomly choosing data (repetition is allowed).
2. Build a decision tree for each dataset, using a random subset of features at each split
3. `For classification:` Use majority voting of trees
4. `For regression:` Take the average prediction of all trees

`Real-life scenario:` Predicting if a loan applicant will `default` based on income, credit score, and employment history.

### Visualization 
![Decision Trees Flowchart](https://raw.githubusercontent.com/tayyab-balti/AI-Internship/master/Images/random-forest-loan-default.svg)

### Pros
- Handles large datasets well.
- Reduces overfitting compared to individual decision trees.
- Works well with both classification and regression tasks.

### Cons
- More complex and slower than a single decision tree.
- Difficult to interpret due to multiple trees.

