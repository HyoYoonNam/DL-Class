Feature selection using **correlation** and **decision tree feature importance** are both valid techniques, but they serve different purposes and may be more appropriate in different situations.

### Correlation
- **When to use**: Correlation is useful when you're dealing with continuous variables and want to understand linear relationships between features. For example, in regression tasks, correlation can help identify highly correlated features that may provide redundant information. It works well when you expect simple linear relationships between variables.
- **Limitations**: Correlation only measures linear relationships, so it won't capture non-linear interactions between features. Additionally, it doesn't take into account the target variable.

### Feature Importance with Decision Trees
- **When to use**: Decision tree-based models like **Random Forests** or **Gradient Boosting Machines (GBM)** provide a more robust way to assess feature importance, especially when the relationships between features and the target are non-linear or complex. These models consider the impact of features on the prediction, making them more suitable for determining how much each feature contributes to the outcome.
- **Advantages**: Decision tree models can handle both continuous and categorical variables, capture non-linear interactions, and focus on the target variable's influence directly, making them more powerful for many real-world tasks.
- **Limitations**: The importance scores can be biased towards features with more categories or higher cardinality, so careful interpretation is needed.

### Conclusion
Using a **decision tree model for feature importance** is generally more appropriate than relying solely on correlation, especially in more complex tasks or when dealing with non-linear relationships. However, if you're looking for quick insights into simple linear dependencies between features, correlation can be useful as a first step.

In practice, you can often combine both techniques: start with correlation to remove highly correlated features, and then use decision tree feature importance to refine your selection.