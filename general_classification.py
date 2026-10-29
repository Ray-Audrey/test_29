
# =====================================================
# 1. LOAD DATA
# =====================================================
import pandas as pd
import numpy as np

df = pd.read_csv("../practice/Cellphone.csv")

target = "YOUR_CLASS_COLUMN"   # CHANGE THIS
id_cols = ["Record ID"]        # CHANGE if needed
RANDOM_STATE = 42

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

df = df.drop_duplicates().reset_index(drop=True)

# =====================================================
# 2. EDA
# =====================================================
import seaborn as sns
import matplotlib.pyplot as plt

display(df.head())
df.info()
display(df.describe(include="all"))

print("Missing values:\n", df.isnull().sum())
print("Skewness:\n", df.skew(numeric_only=True))

# Target distribution: classification
print("\nTarget counts:")
print(df[target].value_counts())

print("\nTarget percentages:")
print(df[target].value_counts(normalize=True) * 100)

sns.countplot(data=df, x=target)
plt.title("Target Class Distribution")
plt.xticks(rotation=45)
plt.show()

# Split features and target; remove ID
X = df.drop(columns=[target] + id_cols, errors="ignore")
y = df[target]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)
print("Number of classes:", y.nunique())

# Predictor distributions
X[num_cols].hist(figsize=(10, 6))
plt.tight_layout()
plt.show()

# Predictor outlier plots
for col in num_cols:
    sns.boxplot(x=X[col])
    plt.title(col)
    plt.show()

# Correlation matrix
print("Correlation Matrix:")
display(df.corr(numeric_only=True))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# =====================================================
# 3. TRAIN-TEST SPLIT
# =====================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# stratify=y preserves class proportions in both sets

# =====================================================
# 4. OUTLIER ANALYSIS (IQR)
# =====================================================
for col in num_cols:
    q1 = X_train[col].quantile(0.25)
    q3 = X_train[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = (
        (X_train[col] < lower_bound) |
        (X_train[col] > upper_bound)
    ).sum()

    print(col, "outliers:", count)

    # If asked to cap outliers:
    # X_train[col] = X_train[col].clip(
    #     lower_bound, upper_bound
    # )

print("Inspect outliers; retain genuine observations.")

# =====================================================
# 5. PREPROCESSING + PCA + MODELS
# =====================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

# Categorical pipeline
cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore",sparse_output=False)
)

# Without PCA
prep_nopca = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    ), num_cols),

    ("cat", cat_pipe, cat_cols)
])

# With PCA: numerical features only
prep_pca = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=0.95)
    ), num_cols),

    ("cat", cat_pipe, cat_cols)
])

# Step name = "model"
m1 = Pipeline([
    ("prep", prep_nopca),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

m2 = Pipeline([
    ("prep", prep_pca),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

# =====================================================
# 6. SAME 5-FOLD CV + INNER HYPERPARAMETER TUNING
# =====================================================

import time

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

grid = {
    "model__max_depth": [None, 3, 5, 7, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__criterion": ["gini", "entropy"]
}

# ONE StratifiedKFold object
kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rows = []

for k, (tr, va) in enumerate(
    kf.split(X_train, y_train), 1
):

    row = {"fold": k}

    for name, m in [
        ("nopca", m1),
        ("pca", m2)
    ]:

        gs = GridSearchCV(
            m,
            grid,
            cv=3,
            n_jobs=-1,
            scoring="accuracy"
        )

        t0 = time.time()

        gs.fit(
            X_train.iloc[tr],
            y_train.iloc[tr]
        )

        row[name + "_time"] = (
            time.time() - t0
        )

        p_tr = gs.predict(
            X_train.iloc[tr]
        )

        p_va = gs.predict(
            X_train.iloc[va]
        )

        row[name + "_train_acc"] = (
            accuracy_score(
                y_train.iloc[tr],
                p_tr
            )
        )

        row[name + "_acc"] = (
            accuracy_score(
                y_train.iloc[va],
                p_va
            )
        )

        row[name + "_precision"] = (
            precision_score(
                y_train.iloc[va],
                p_va,
                average="weighted",
                zero_division=0
            )
        )

        row[name + "_recall"] = (
            recall_score(
                y_train.iloc[va],
                p_va,
                average="weighted",
                zero_division=0
            )
        )

        row[name + "_f1"] = (
            f1_score(
                y_train.iloc[va],
                p_va,
                average="weighted",
                zero_division=0
            )
        )

        row[name + "_params"] = (
            gs.best_params_
        )

    rows.append(row)

res = pd.DataFrame(rows)

res["diff"] = (
    res["pca_acc"] -
    res["nopca_acc"]
)

print(
    res[
        [
            "fold",
            "nopca_acc",
            "pca_acc",
            "diff"
        ]
    ]
)

print(
    res.filter(
        regex="acc|precision|recall|f1"
    ).agg(["mean", "std"])
)

'''# train-validation gap
print(
    "No PCA gap:",
    (
        res["nopca_train_acc"]
        - res["nopca_acc"]
    ).mean()
)

print(
    "PCA gap:",
    (
        res["pca_train_acc"]
        - res["pca_acc"]
    ).mean()
)'''
# =====================================================
# 7. FOLD-WISE ACCURACY COMPARISON
# =====================================================
print("\nFold-wise accuracy comparison:")

res["diff"] = res["pca_acc"] - res["nopca_acc"]

display(res[[
    "fold",
    "nopca_acc",
    "pca_acc",
    "diff"
]])

# Best fold and its parameters (descriptive only)
print(
    "Best fold (No PCA):",
    res.loc[res["nopca_acc"].idxmax(), "fold"]
)

print(
    "Parameters in that fold (No PCA):",
    res.loc[res["nopca_acc"].idxmax(), "nopca_params"]
)

print(
    "Best fold (PCA):",
    res.loc[res["pca_acc"].idxmax(), "fold"]
)

print(
    "Parameters in that fold (PCA):",
    res.loc[res["pca_acc"].idxmax(), "pca_params"]
)

# =====================================================
# 8. MEAN + STANDARD DEVIATION
# =====================================================
print("\nFold-wise metric summary:")

display(
    res.filter(
        regex="acc|precision|recall|f1|time"
    ).agg(["mean", "std"]).T
)

print("\nBest parameters by fold:")
display(res[[
    "fold", "nopca_params", "pca_params"
]])

'''print("\nTrain-validation accuracy gap:")
print("No PCA:", (res["nopca_train_acc"] - res["nopca_acc"]).mean())
print("PCA:", (res["pca_train_acc"] - res["pca_acc"]).mean())'''

# =====================================================
# 9. PAIRED T-TEST ON FOLD ACCURACY
# =====================================================
from scipy.stats import ttest_rel

t, p = ttest_rel(res["pca_acc"], res["nopca_acc"])

print("t =", t, "p =", p)

if p < 0.05:
    print("Reject H0: significant accuracy difference detected.")
else:
    print("Fail to reject H0: no significant difference detected.")

# =====================================================
# 10. PCA DIMENSIONS + EXPLAINED VARIANCE
# =====================================================
Xs = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
).fit_transform(X_train[num_cols])

pca = PCA().fit(Xs)

cum = np.cumsum(pca.explained_variance_ratio_)

n95 = np.argmax(cum >= 0.95) + 1

print("Original numerical dimensions:", Xs.shape[1])
print("Components for at least 95% variance:", n95)

plt.plot(
    range(1, len(cum) + 1),
    cum,
    marker="o"
)
plt.axhline(0.95, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

pca95 = PCA(n_components=0.95).fit(Xs)

print("Components selected:", pca95.n_components_)
print("Variance retained:", pca95.explained_variance_ratio_.sum())

'''# =====================================================
# FINAL MODEL + TEST EVALUATION
# =====================================================
gs1 = GridSearchCV(
    m1,
    grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

gs2 = GridSearchCV(
    m2,
    grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

gs1.fit(X_train, y_train)
gs2.fit(X_train, y_train)

best_model = gs1.best_estimator_ \
    if res["nopca_acc"].mean() >= res["pca_acc"].mean() \
    else gs2.best_estimator_

y_pred = best_model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )
)

print(
    "F1:",
    f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)'''