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
