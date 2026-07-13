
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("cleaned_data.csv")

print(df.head())

print(df.shape)

y_reg = df["MSRP"]
y_clf = (df["MSRP"] > df["MSRP"].median()).astype(int)

print(y_clf.value_counts())

X = df.drop(columns=["MSRP"])

cat_cols = X.select_dtypes(include=["object","category"]).columns

cat_cols

X = pd.get_dummies(
    X,
    columns=cat_cols,
    drop_first=True
)

print(X.shape)

X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

_, _, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

linear = LinearRegression()

linear.fit(
    X_train_scaled,
    y_train_reg
)

y_pred = linear.predict(X_test_scaled)

mse = mean_squared_error(
    y_test_reg,
    y_pred
)

r2 = r2_score(
    y_test_reg,
    y_pred
)

print("MSE :", mse)

print("R2 :", r2)

coef = pd.DataFrame({

    "Feature":X.columns,

    "Coefficient":linear.coef_

})

coef["Absolute"] = coef["Coefficient"].abs()

coef = coef.sort_values(
    "Absolute",
    ascending=False
)

coef.head(10)

coef.head(3)

# Train Ridge Regression Model

ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train_reg)

# Prediction
ridge_pred = ridge.predict(X_test_scaled)

# Evaluation
ridge_mse = mean_squared_error(y_test_reg, ridge_pred)
ridge_r2 = r2_score(y_test_reg, ridge_pred)

print("Ridge MSE :", ridge_mse)
print("Ridge R² :", ridge_r2)

comparison = pd.DataFrame({

    "Model":["Linear Regression","Ridge Regression"],

    "MSE":[mse,ridge_mse],

    "R²":[r2,ridge_r2]

})

comparison

print("Training Class Distribution")

print(y_train_clf.value_counts())

class_ratio = y_train_clf.value_counts(normalize=True)

print(class_ratio)

logistic = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    random_state=42

)

logistic.fit(

    X_train_scaled,

    y_train_clf

)

y_pred = logistic.predict(X_test_scaled)

y_probability = logistic.predict_proba(X_test_scaled)[:,1]

cm = confusion_matrix(

    y_test_clf,

    y_pred

)

cm

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    cmap="Blues",

    fmt="d"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

print(

classification_report(

    y_test_clf,

    y_pred

)

)

accuracy = logistic.score(

    X_test_scaled,

    y_test_clf

)

precision = precision_score(

    y_test_clf,

    y_pred

)

recall = recall_score(

    y_test_clf,

    y_pred

)

f1 = f1_score(

    y_test_clf,

    y_pred

)

print("Accuracy :",accuracy)

print("Precision :",precision)

print("Recall :",recall)

print("F1 Score :",f1)

fpr,tpr,threshold = roc_curve(

    y_test_clf,

    y_probability

)

auc = roc_auc_score(

    y_test_clf,

    y_probability

)

plt.figure(figsize=(8,6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {auc:.3f}"

)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

"""Markdown
ROC Interpretation

An ROC curve closer to the upper-left corner indicates better classification performance.

The AUC measures the probability that the classifier ranks a randomly chosen positive sample higher than a randomly chosen negative sample.

AUC = 1.0 → Perfect classifier
AUC = 0.5 → Random guessing
"""

thresholds = np.arange(0.30,0.71,0.10)

results=[]

for t in thresholds:

    pred=(y_probability>=t).astype(int)

    results.append([

        t,

        precision_score(y_test_clf,pred),

        recall_score(y_test_clf,pred),

        f1_score(y_test_clf,pred)

    ])

threshold_table=pd.DataFrame(

    results,

    columns=[

        "Threshold",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

threshold_table

best = threshold_table.loc[

    threshold_table["F1 Score"].idxmax()

]

best

"""Markdown
Precision Formula
Precision=
(TP+FP)/
TP
	​

Recall Formula
Recall=
(TP+FN)/
TP
	​


Higher thresholds increase Precision but decrease Recall.

Lower thresholds increase Recall but may introduce more False Positives.

The threshold with the highest F1-score provides the best balance between Precision and Recall.
"""

regularized = LogisticRegression(

    C=0.01,

    max_iter=1000,

    random_state=42

)

regularized.fit(

    X_train_scaled,

    y_train_clf

)

reg_probability = regularized.predict_proba(

    X_test_scaled

)[:,1]

reg_prediction = regularized.predict(

    X_test_scaled

)

comparison=pd.DataFrame({

"Model":["C = 1.0","C = 0.01"],

"Precision":[

precision_score(y_test_clf,y_pred),

precision_score(y_test_clf,reg_prediction)

],

"Recall":[

recall_score(y_test_clf,y_pred),

recall_score(y_test_clf,reg_prediction)

],

"AUC":[

roc_auc_score(y_test_clf,y_probability),

roc_auc_score(y_test_clf,reg_probability)

]

})

comparison

"""Markdown
What does C control?

The parameter C is the inverse of regularization strength.

Larger C → weaker regularization.
Smaller C → stronger regularization.

Reducing C shrinks coefficients more aggressively, helping prevent overfitting but potentially reducing predictive performance.
"""

auc_difference=[]

for i in range(500):

    index=np.random.choice(

        len(y_test_clf),

        len(y_test_clf),

        replace=True

    )

    auc1=roc_auc_score(

        y_test_clf.iloc[index],

        y_probability[index]

    )

    auc2=roc_auc_score(

        y_test_clf.iloc[index],

        reg_probability[index]

    )

    auc_difference.append(

        auc1-auc2

    )

mean_difference=np.mean(auc_difference)

lower=np.percentile(

    auc_difference,

    2.5

)

upper=np.percentile(

    auc_difference,

    97.5

)

print("Mean Difference :",mean_difference)

print("95% Confidence Interval")

print(lower)

print(upper)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Baseline Decision Tree

dt_default = DecisionTreeClassifier(random_state=42)

dt_default.fit(X_train_scaled, y_train_clf)

train_pred_default = dt_default.predict(X_train_scaled)

train_acc_default = accuracy_score(
    y_train_clf,
    train_pred_default
)


test_pred_default = dt_default.predict(X_test_scaled)

test_acc_default = accuracy_score(
    y_test_clf,
    test_pred_default
)

print("Baseline Decision Tree")

print("----------------------")

print("Training Accuracy :", train_acc_default)

print("Testing Accuracy :", test_acc_default)

controlled_tree = DecisionTreeClassifier(

    max_depth=5,

    min_samples_split=20,

    random_state=42

)

controlled_tree.fit(

    X_train_scaled,

    y_train_clf

)

controlled_train = controlled_tree.predict(X_train_scaled)

controlled_train_acc = accuracy_score(

    y_train_clf,

    controlled_train

)

controlled_test = controlled_tree.predict(X_test_scaled)

controlled_test_acc = accuracy_score(

    y_test_clf,

    controlled_test

)
print("Controlled Decision Tree")

print("--------------------------")

print("Training Accuracy :", controlled_train_acc)

print("Testing Accuracy :", controlled_test_acc)

comparison = pd.DataFrame({

    "Model":[

        "Default Tree",

        "Controlled Tree"

    ],

    "Training Accuracy":[

        train_acc_default,

        controlled_train_acc

    ],

    "Testing Accuracy":[

        test_acc_default,

        controlled_test_acc

    ]

})

comparison

gini_tree = DecisionTreeClassifier(

    criterion="gini",

    max_depth=5,

    random_state=42

)

gini_tree.fit(

    X_train_scaled,

    y_train_clf

)

gini_accuracy = accuracy_score(

    y_test_clf,

    gini_tree.predict(X_test_scaled)

)

entropy_tree = DecisionTreeClassifier(

    criterion="entropy",

    max_depth=5,

    random_state=42

)

entropy_tree.fit(

    X_train_scaled,

    y_train_clf

)

entropy_accuracy = accuracy_score(

    y_test_clf,

    entropy_tree.predict(X_test_scaled)

)

criteria = pd.DataFrame({

    "Criterion":[

        "Gini",

        "Entropy"

    ],

    "Testing Accuracy":[

        gini_accuracy,

        entropy_accuracy

    ]

})

criteria

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train_clf)

# Training Accuracy
rf_train_pred = rf_model.predict(X_train_scaled)

rf_train_acc = accuracy_score(
    y_train_clf,
    rf_train_pred
)

# Test Accuracy
rf_test_pred = rf_model.predict(X_test_scaled)

rf_test_acc = accuracy_score(
    y_test_clf,
    rf_test_pred
)

# ROC-AUC
rf_prob = rf_model.predict_proba(X_test_scaled)[:,1]

rf_auc = roc_auc_score(
    y_test_clf,
    rf_prob
)

print("Random Forest Training Accuracy :", rf_train_acc)
print("Random Forest Testing Accuracy :", rf_test_acc)
print("Random Forest ROC-AUC :", rf_auc)
importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.head(5)

top5 = importance.head(5)

print(top5)

plt.figure(figsize=(8,5))

plt.barh(
    top5["Feature"],
    top5["Importance"]
)

plt.xlabel("Importance")

plt.title("Top 5 Feature Importance - Random Forest")

plt.gca().invert_yaxis()

plt.show()

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42

)

gb_model.fit(
    X_train_scaled,
    y_train_clf
)

gb_train_pred = gb_model.predict(X_train_scaled)

gb_test_pred = gb_model.predict(X_test_scaled)

gb_train_acc = accuracy_score(
    y_train_clf,
    gb_train_pred
)

gb_test_acc = accuracy_score(
    y_test_clf,
    gb_test_pred
)

gb_prob = gb_model.predict_proba(X_test_scaled)[:,1]

gb_auc = roc_auc_score(
    y_test_clf,
    gb_prob
)

print("Gradient Boosting Training Accuracy :", gb_train_acc)

print("Gradient Boosting Testing Accuracy :", gb_test_acc)

print("Gradient Boosting ROC-AUC :", gb_auc)

ensemble_results = pd.DataFrame({

    "Model":[
        "Random Forest",
        "Gradient Boosting"
    ],

    "Training Accuracy":[
        rf_train_acc,
        gb_train_acc
    ],

    "Testing Accuracy":[
        rf_test_acc,
        gb_test_acc
    ],

    "ROC-AUC":[
        rf_auc,
        gb_auc
    ]

})

ensemble_results

lowest5 = importance.tail(5)

lowest5

remove_features = lowest5["Feature"].tolist()

X_train_reduced = X_train.drop(columns=remove_features)

X_test_reduced = X_test.drop(columns=remove_features)

scaler_reduced = StandardScaler()

X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)

X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)


rf_reduced = RandomForestClassifier(

    n_estimators=100,

    max_depth=10,

    random_state=42

)

rf_reduced.fit(
    X_train_reduced_scaled,
    y_train_clf
)

full_auc = roc_auc_score(

    y_test_clf,

    rf_model.predict_proba(X_test_scaled)[:,1]

)

reduced_auc = roc_auc_score(

    y_test_clf,

    rf_reduced.predict_proba(X_test_reduced_scaled)[:,1]

)

comparison = pd.DataFrame({

    "Model":[

        "All Features",

        "Reduced Features"

    ],

    "ROC-AUC":[

        full_auc,

        reduced_auc

    ]

})

comparison

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# ----------------------------------------------------------
# Stratified K-Fold
# ----------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("5-Fold Cross Validation Created Successfully")


# ----------------------------------------------------------
# Models
# ----------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

gradient_boost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# ----------------------------------------------------------
# Cross Validation
# ----------------------------------------------------------

models = {

    "Logistic Regression": logistic_model,

    "Decision Tree": decision_tree,

    "Random Forest": random_forest,

    "Gradient Boosting": gradient_boost

}

results = []

for name, model in models.items():

    scores = cross_val_score(

        model,

        X_train_scaled,

        y_train_clf,

        cv=cv,

        scoring="roc_auc"

    )

    results.append({

        "Model": name,

        "Mean AUC": scores.mean(),

        "Std AUC": scores.std()

    })

cv_results = pd.DataFrame(results)

print("\nCross Validation Results\n")

print(cv_results)

# ----------------------------------------------------------
# Pipeline
# ----------------------------------------------------------

pipeline = make_pipeline(

    SimpleImputer(strategy="median"),

    StandardScaler(),

    RandomForestClassifier(random_state=42)

)

print("\nPipeline Created Successfully")

# ----------------------------------------------------------
# Parameter Grid
# ----------------------------------------------------------

param_grid = {

    "randomforestclassifier__n_estimators":[50,100,200],

    "randomforestclassifier__max_depth":[5,10,None],

    "randomforestclassifier__min_samples_leaf":[1,5]

}

# ----------------------------------------------------------
# Grid Search
# ----------------------------------------------------------

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=cv,

    n_jobs=-1,

    verbose=1

)

# IMPORTANT
# Use ORIGINAL training data (not scaled)
grid_search.fit(

    X_train,

    y_train_clf

)

print("\nGrid Search Completed Successfully")

# ----------------------------------------------------------
# Best Parameters
# ----------------------------------------------------------

print("\nBest Parameters")

print(grid_search.best_params_)

print("\nBest Cross Validation Score")

print(grid_search.best_score_)

# ----------------------------------------------------------
# Total Models Evaluated
# ----------------------------------------------------------

total_models = (

    len(param_grid["randomforestclassifier__n_estimators"])

    *

    len(param_grid["randomforestclassifier__max_depth"])

    *

    len(param_grid["randomforestclassifier__min_samples_leaf"])

)

print("\nTotal Parameter Combinations :", total_models)

print("Total Models Trained :", total_models * 5)

# ----------------------------------------------------------
# Best Pipeline
# ----------------------------------------------------------

best_pipeline = grid_search.best_estimator_

print("\nBest Pipeline")

print(best_pipeline)

# ----------------------------------------------------------
# Summary Table
# ----------------------------------------------------------

summary = cv_results.copy()

summary["GridSearch Best Score"] = ""

summary.loc[
    summary["Model"]=="Random Forest",
    "GridSearch Best Score"
] = round(grid_search.best_score_,4)

print("\nSummary")

print(summary)

# ============================================================
# PART 3.4
# Manual Learning Curve
# Model Serialization
# Final Model Comparison
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------
# Manual Learning Curve
# ------------------------------------------------------------

fractions = [0.2,0.4,0.6,0.8,1.0]

learning_curve = []

print("Training Learning Curve...\n")

for fraction in fractions:

    rows = int(fraction * len(X_train))

    X_subset = X_train.iloc[:rows]

    y_subset = y_train_clf.iloc[:rows]

    # Train Best Pipeline
    best_pipeline.fit(

        X_subset,

        y_subset

    )

    # Training AUC

    train_probability = best_pipeline.predict_proba(

        X_subset

    )[:,1]

    train_auc = roc_auc_score(

        y_subset,

        train_probability

    )

    # Test AUC

    test_probability = best_pipeline.predict_proba(

        X_test

    )[:,1]

    test_auc = roc_auc_score(

        y_test_clf,

        test_probability

    )

    learning_curve.append({

        "Training Fraction":fraction,

        "Training AUC":train_auc,

        "Test AUC":test_auc

    })

learning_table = pd.DataFrame(

    learning_curve

)

print(learning_table)

# ------------------------------------------------------------
# Save Best Model
# ------------------------------------------------------------

joblib.dump(

    best_pipeline,

    "best_model.pkl"

)

print("\nModel saved successfully.")

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

loaded_model = joblib.load(

    "best_model.pkl"

)

print("Model loaded successfully.")

# ------------------------------------------------------------
# Create Two Sample Cars
# ------------------------------------------------------------

sample_rows = X_test.iloc[:2].copy()

print("\nSample Cars")

print(sample_rows)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

prediction = loaded_model.predict(

    sample_rows

)

probability = loaded_model.predict_proba(

    sample_rows

)

print("\nPredicted Class")

print(prediction)

print("\nPrediction Probability")

print(probability)

final_models = pd.DataFrame({

"Model":[

"Logistic Regression",

"Decision Tree",

"Random Forest",

"Gradient Boosting"

],

"5-Fold Mean AUC":[

cv_results.loc[
cv_results.Model=="Logistic Regression",
"Mean AUC"].values[0],

cv_results.loc[
cv_results.Model=="Decision Tree",
"Mean AUC"].values[0],

cv_results.loc[
cv_results.Model=="Random Forest",
"Mean AUC"].values[0],

cv_results.loc[
cv_results.Model=="Gradient Boosting",
"Mean AUC"].values[0]

],

"5-Fold Std AUC":[

cv_results.loc[
cv_results.Model=="Logistic Regression",
"Std AUC"].values[0],

cv_results.loc[
cv_results.Model=="Decision Tree",
"Std AUC"].values[0],

cv_results.loc[
cv_results.Model=="Random Forest",
"Std AUC"].values[0],

cv_results.loc[
cv_results.Model=="Gradient Boosting",
"Std AUC"].values[0]

]

})

print("\nFinal Comparison")

print(final_models)

best = final_models.sort_values(

    by="5-Fold Mean AUC",

    ascending=False

)

print("\nRecommended Model")

print(best.head(1))

print("\nPart 3 Completed Successfully")