# =====================================================
# 1. LOAD DATA
# =====================================================
import pandas as pd
import numpy as np

df = pd.read_csv("../practice/Cellphone.csv")

target = "Remaining Useful Life Hours"  # change if needed
id_cols = ["Record ID"]                # change if needed
RANDOM_STATE = 42

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Remove duplicate rows
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

df[target].hist()
plt.title("Target Distribution")
plt.show()

sns.boxplot(x=df[target])
plt.title("Target Boxplot")
plt.show()

# Split features and target; remove unique ID
X = df.drop(columns=[target] + id_cols, errors="ignore")
y = df[target]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

X[num_cols].hist(figsize=(10, 6))
plt.tight_layout()
plt.show()

# Predictor outlier plots
for col in num_cols:
    sns.boxplot(x=X[col])
    plt.title(col)
    plt.show()

# Correlation heatmap
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
    X, y, test_size=0.2, random_state=RANDOM_STATE
)  # use stratify=<y_or_None> only for classification tasks

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
    #Winsorization / Capping (if asked to treat outliers)
    #    X_train[col] = X_train[col].clip(lower_bound,upper_bound)
print("Inspect outliers; retain genuine observations.")

# =====================================================
# 5. PREPROCESSING + PCA
# =====================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVR

# Categorical pipeline
cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore",sparse_output=False)
)

# Without PCA: impute -> scale numerical features
prep_nopca = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    ), num_cols),
    ("cat", cat_pipe, cat_cols)
])

# With PCA: impute -> scale -> PCA numerical features only
prep_pca = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=0.95)
    ), num_cols),

    ("cat", cat_pipe, cat_cols)
])

# Cheatsheet step names: prep, svr
m1 = Pipeline([
    ("prep", prep_nopca),
    ("svr", SVR(kernel="rbf"))
])

m2 = Pipeline([
    ("prep", prep_pca),
    ("svr", SVR(kernel="rbf"))
])
'''**svr=rbf
grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.1],
    "svr__epsilon": [0.01, 0.1, 0.5]
}
What changes for other algorithms?
KNN
Python
m1 = Pipeline([
("prep", prep_nopca),
("model", KNeighborsRegressor())
])
 
m2 = Pipeline([
("prep", prep_pca),
("model", KNeighborsRegressor())
])

Grid:

Python
grid = {
"model__n_neighbors": [3,5,7],
"model__weights": ["uniform","distance"]
}
Decision Tree
Python
m1 = Pipeline([
("prep", prep_nopca),
("model", DecisionTreeRegressor())
])
 
m2 = Pipeline([
("prep", prep_pca),
("model", DecisionTreeRegressor())
])

Grid:

Python
grid = {
"model__max_depth":[3,5,10,None]
}
Random Forest
Python
m1 = Pipeline([
("prep", prep_nopca),
("model", RandomForestRegressor())
])
 
m2 = Pipeline([
("prep", prep_pca),
("model", RandomForestRegressor())
])

Grid:

Python
grid = {
"model__n_estimators":[100,200]'''
# =====================================================
# 6. SAME 5-FOLD CV + INNER HYPERPARAMETER TUNING
# =====================================================
import time

from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.1],
    "svr__epsilon": [0.01, 0.1, 0.5]
}

# ONE KFold object -> identical folds
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

rows = []

for k, (tr, va) in enumerate(kf.split(X_train), 1):

    row = {"fold": k}
'''for three models
 m1 = Pipeline([
    ("prep", prep_nopca),
    ("model", RandomForestRegressor())
])
m2 = Pipeline([
    ("prep", prep_pca),
    ("model", RandomForestRegressor())
])
m3 = Pipeline([
    ("prep", prep_pca),
    ("model", DecisionTreeRegressor())
])grid1 = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
    "model__criterion": ["squared_error"],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_split": [2, 5, 10]
}
grid2 = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
    "model__criterion": ["squared_error"],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_split": [2, 5, 10]
}
grid3 = {
    "model__max_depth": [None, 3, 5, 7, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}for name, m, grid in [
    ("nopca_rf", m1, grid1),
    ("pca_rf", m2, grid2),
    ("pca_dt", m3, grid3)
]:'''
    for name, m in [
        ("nopca", m1),
        ("pca", m2)
    ]:

        gs = GridSearchCV(
            estimator=m,
            param_grid=grid,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        t0 = time.time()

        gs.fit(
            X_train.iloc[tr],
            y_train.iloc[tr]
        )

        row[name + "_time"] = time.time() - t0

        p_tr = gs.predict(
            X_train.iloc[tr]
        )

        p_va = gs.predict(
            X_train.iloc[va]
        )

        row[name + "_train_rmse"] = (
            mean_squared_error(
                y_train.iloc[tr],
                p_tr
            ) ** 0.5
        )

        row[name + "_rmse"] = (
            mean_squared_error(
                y_train.iloc[va],
                p_va
            ) ** 0.5
        )

        row[name + "_mae"] = (
            mean_absolute_error(
                y_train.iloc[va],
                p_va
            )
        )

        row[name + "_r2"] = (
            r2_score(
                y_train.iloc[va],
                p_va
            )
        )

        row[name + "_mape"] = (
            mean_absolute_percentage_error(
                y_train.iloc[va],
                p_va
            ) * 100
        )

        row[name + "_params"] = gs.best_params_
    rows.append(row)

res = pd.DataFrame(rows)

# =====================================================
# 7. FOLD-WISE RMSE COMPARISON
# =====================================================
print("\nFold-wise rmse comparison:")
res["diff"] = (
    res["pca_rmse"] -
    res["nopca_rmse"]
)

print(
    res[
        ["fold",
         "nopca_rmse",
         "pca_rmse",
         "diff"]
    ]
)
'''for three models:
print("\nFold-wise RMSE comparison:")

print(
    res[
        [
            "fold",
            "nopca_rf_rmse",
            "pca_rf_rmse",
            "pca_dt_rmse"
        ]
    ]
)'''
# =====================================================
# BEST FOLD Best Parameters
# =====================================================
print(
    "Best fold (No PCA):",
    res.loc[
        res["nopca_rmse"].idxmin(),
        "fold"
    ]
)
print(
    "Best parameters (No PCA):",
    res.loc[
        res["nopca_rmse"].idxmin(),
        "nopca_params"
    ]
)
print(
    "Best fold (PCA):",
    res.loc[
        res["pca_rmse"].idxmin(),
        "fold"
    ]
)

print(
    "Best parameters (PCA):",
    res.loc[
        res["pca_rmse"].idxmin(),
        "pca_params"
    ]
)
# =====================================================
# 8. MEAN + STANDARD DEVIATION
# =====================================================

print("\nFold-wise metric summary:")

print(res.filter(
    regex="rmse|mae|r2|mape|time"
).agg(["mean", "std"]).T)

print("\nBest parameters by fold:")
display(res[[
    "fold", "nopca_params", "pca_params"
]])
'''For 3 models, change to:
Python
display(res[[
"fold",
"nopca_rf_params",
"pca_rf_params",
"pca_dt_params"
]])'''
# =====================================================
# 9. PAIRED T-TEST ON FOLD RMSE
# =====================================================
from scipy.stats import ttest_rel

t, p = ttest_rel(res["pca_rmse"], res["nopca_rmse"])

print("t =", t, "p =", p)

if p < 0.05:
    print("Reject H0: significant RMSE difference detected.")
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

cum = np.cumsum(
    pca.explained_variance_ratio_
)

n95 = np.argmax(
    cum >= 0.95
) + 1

print(
    Xs.shape[1],
    "->",
    n95
)

plt.plot(
    range(1, len(cum) + 1),
    cum,
    marker="o"
)

plt.axhline(
    0.95,
    linestyle="--"
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()
pca95 = PCA(
    n_components=0.95
).fit(Xs)

print(
    "Components selected:",
    pca95.n_components_
)

print(
    "Variance retained:",
    pca95.explained_variance_ratio_.sum()
)
'''# =====================================================
# 11. FINAL TEST-SET EVALUATION
# =====================================================

# Retrain on full training data

gs1 = GridSearchCV(
    m1,
    grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

gs2 = GridSearchCV(
    m2,
    grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

gs1.fit(X_train, y_train)
gs2.fit(X_train, y_train)

print(
    "No PCA Best Parameters:",
    gs1.best_params_
)

print(
    "PCA Best Parameters:",
    gs2.best_params_
)

# Select better model based on mean CV RMSE

if res["nopca_rmse"].mean() <= res["pca_rmse"].mean():
    best_model = gs1.best_estimator_
    best_name = "SVR without PCA"
else:
    best_model = gs2.best_estimator_
    best_name = "SVR with PCA"

print(
    "\nSelected Model:",
    best_name
)

# Test-set predictions

y_pred = best_model.predict(X_test)

# Test metrics

print(
    "Test RMSE:",
    mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5
)

print(
    "Test MAE:",
    mean_absolute_error(
        y_test,
        y_pred
    )
)

print(
    "Test R²:",
    r2_score(
        y_test,
        y_pred
    )
)

print(
    "Test MAPE:",
    mean_absolute_percentage_error(
        y_test,
        y_pred
    ) * 100
)'''