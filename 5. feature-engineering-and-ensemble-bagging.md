## Feature Engineering
- Using intuition to design new features, by transforming or combining original features.

## Ensemble Learning
- It uses multiple learning algorithms to obtain better predictive performance.
- Giving dataset to multiple models (KNN, SVM, Logistic Regression, etc.)
- Types: Bootstrap aggregating, Boosting, Stacking, Voting

## Bagging/Bootstrap Aggregating
- It is an ensemble machine learning technique designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also helps reduce overfitting.

### How it works:
- **Bootstrap Sampling**: Multiple subsets of the original dataset are created by randomly sampling with replacement. This means some observations may be repeated in each subset.
- **Model Training**: A separate model is trained on each bootstrap sample.
- **Predictions**: Each model makes its own prediction on new data.
- **Aggregation**: The predictions from all models are combined to create a final prediction. This could be through majority voting (for classification) or averaging (for regression).

- `Random Forest` is a popular algorithm that uses bagging with decision trees.

### How Weights are Increased in Boosting:
1. **Initialize weights**: If there are 5 total rows, then each row has a weight of \( \frac{1}{5} \).
2. **Train weak learner 1 (model)**.
3. **Calculate total error**: Sum the errors of misclassified rows.
4. **Calculate weak learner weight (α)**: 1/2 ln(1-total error/total error)
5. **Update instance weights**: old weight * e^α
6. **Repeat from step 2** until the desired number of weak learners is trained.
