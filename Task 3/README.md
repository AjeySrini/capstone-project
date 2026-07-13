Advanced Machine Learning (Part 3)

In this part, I trained different machine learning models and compared their performance to find the best model.

Pipeline

I created a Scikit-learn pipeline using:

SimpleImputer
StandardScaler
RandomForestClassifier

This pipeline automatically handles preprocessing and model training.

Hyperparameter Tuning

I used GridSearchCV to find the best Random Forest parameters.

Parameters tested:

n_estimators: 50, 100, 200
max_depth: 5, 10, None
min_samples_leaf: 1, 5

Best Parameters:

n_estimators: (Output)
max_depth: (Output)
min_samples_leaf: (Output)

Best ROC-AUC Score: (Output)

Cross Validation

I used 5-fold cross-validation to evaluate the models.

Model	Mean AUC
Logistic Regression	(Output)
Decision Tree	(Output)
Random Forest	(Output)
Gradient Boosting	(Output)

Cross-validation gives a better estimate of model performance because the model is tested on different data splits.

Feature Importance

I used the Random Forest model to find the top 5 important features. These features had the biggest impact on the predictions.

Feature Ablation

I removed the 5 least important features and trained the model again.

Full Model AUC: (Output)
Reduced Model AUC: (Output)

If both AUC values are similar, the removed features were not very important. This also makes the model simpler and faster.

Learning Curve
Training Data	Training AUC	Test AUC
20%	(Output)	(Output)
40%	(Output)	(Output)
60%	(Output)	(Output)
80%	(Output)	(Output)
100%	(Output)	(Output)

The learning curve shows how the model performance changes as more training data is used.

Best Model

After comparing all models, (Best Model Name) gave the best performance and is recommended for prediction.

Model Saving

The best model was saved as best_model.pkl using Joblib. It was loaded successfully and used to predict new sample data.

Conclusion

This project includes Decision Tree, Random Forest, Gradient Boosting, GridSearchCV, feature importance, feature ablation, learning curve analysis, and model serialization. After comparing all models, the best-performing model was selected for future predictions.