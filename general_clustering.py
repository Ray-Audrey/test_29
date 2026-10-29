
# =====================================================
# 1. LOAD DATA
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv("../practice/Cellphone.csv")

id_cols = ["Record ID"]       # change according to dataset
drop_cols = []                # columns irrelevant to clustering
RANDOM_STATE = 42
K_VALUES = [2, 3, 4, 5, 6]

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

df = df.drop_duplicates().reset_index(drop=True)

# Remove ID and irrelevant columns
df = df.drop(columns=id_cols + drop_cols, errors="ignore")

# =====================================================
# 2. EDA
# =====================================================
display(df.head())
df.info()
display(df.describe(include="all"))

print("Missing values:\n", df.isnull().sum())
print("Skewness:\n", df.skew(numeric_only=True))

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# Numerical feature distributions
if len(num_cols) > 0:
    df[num_cols].hist(figsize=(10, 6))
    plt.tight_layout()
    plt.show()

# Numerical feature outliers
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Correlation matrix
if len(num_cols) > 1:
    print("Correlation Matrix:")
    display(df[num_cols].corr())

    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Correlation Heatmap")
    plt.show()

# =====================================================
# 3. SELECT FEATURES
# =====================================================
# Unsupervised learning has NO target variable y.
# All selected columns are used as clustering features.

X = df.copy()

# Remove constant columns
constant_cols = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]
X = X.drop(columns=constant_cols)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

print("Features used:", X.columns.tolist())
print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# =====================================================
# 4. OUTLIER ANALYSIS (IQR)
# =====================================================
for col in num_cols:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = (
        (X[col] < lower_bound) |
        (X[col] > upper_bound)
    ).sum()

    print(col, "outliers:", count)

# =====================================================
# 5. PREPROCESSING + PCA
# =====================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
)

# Without PCA
prep_nopca = ColumnTransformer([
    ("num",
     make_pipeline(
         SimpleImputer(strategy="median"),
         StandardScaler()
     ),
     num_cols),

    ("cat",
     cat_pipe,
     cat_cols)
])

# With PCA
prep_pca = make_pipeline(
    prep_nopca,
    PCA(n_components=0.95)
)

X_nopca = prep_nopca.fit_transform(X)
X_pca = prep_pca.fit_transform(X)

print(
    "Original processed dimensions:",
    X_nopca.shape[1]
)

print(
    "PCA dimensions:",
    X_pca.shape[1]
)

print(
    "Variance retained:",
    PCA(n_components=0.95)
    .fit(X_nopca)
    .explained_variance_ratio_
    .sum()
)
# =====================================================
# 6. K-MEANS CLUSTERING
# =====================================================

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

results = []

for name, data in [
    ("No PCA", X_nopca),
    ("PCA", X_pca)
]:

    for k in K_VALUES:

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE
        )

        labels = kmeans.fit_predict(data)

        results.append({
            "representation": name,
            "k": k,
            "silhouette":
                silhouette_score(
                    data,
                    labels
                ),
            "calinski_harabasz":
                calinski_harabasz_score(
                    data,
                    labels
                ),
            "davies_bouldin":
                davies_bouldin_score(
                    data,
                    labels
                ),
            "inertia":
                kmeans.inertia_
        })

res = pd.DataFrame(results)

print("\nClustering Results:")
display(res)
# =====================================================
# 7. BEST MODEL SELECTION
# =====================================================

best = res.loc[
    res["silhouette"].idxmax()
]

print(
    "\nBest Representation:",
    best["representation"]
)

print(
    "Best k:",
    int(best["k"])
)

print(
    "Best Silhouette:",
    best["silhouette"]
)

print(
    "Calinski-Harabasz:",
    best["calinski_harabasz"]
)

print(
    "Davies-Bouldin:",
    best["davies_bouldin"]
)
# =====================================================
# 11. PAIRED T-TEST ON SILHOUETTE SCORES
# =====================================================
from scipy.stats import ttest_rel

nopca_sil = (
    res[res["representation"] == "No PCA"]
    .sort_values("k")["silhouette"]
    .values
)

pca_sil = (
    res[res["representation"] == "PCA"]
    .sort_values("k")["silhouette"]
    .values
)

t, p = ttest_rel(pca_sil, nopca_sil)

print("t =", t)
print("p =", p)

if p < 0.05:
    print("Reject H0: Significant difference detected.")
else:
    print("Fail to reject H0: No significant difference detected.")
    
# =====================================================
# 8. FINAL K-MEANS MODEL
# =====================================================

final_data = (
    X_pca
    if best["representation"] == "PCA"
    else X_nopca
)

final_model = KMeans(
    n_clusters=int(best["k"]),
    random_state=RANDOM_STATE
)

final_labels = final_model.fit_predict(
    final_data
)

print(
    "\nFinal Silhouette:",
    silhouette_score(
        final_data,
        final_labels
    )
)

print(
    "Final Calinski-Harabasz:",
    calinski_harabasz_score(
        final_data,
        final_labels
    )
)
print(
    "Final Davies-Bouldin:",
    davies_bouldin_score(
        final_data,
        final_labels
    )
)

print(
    "Final Inertia:",
    final_model.inertia_
)
# =====================================================
# 9. CLUSTER VISUALIZATION
# =====================================================

X_plot = PCA(
    n_components=2
).fit_transform(final_data)

sns.scatterplot(
    x=X_plot[:, 0],
    y=X_plot[:, 1],
    hue=final_labels,
    palette="deep"
)

plt.title(
    f"KMeans Clusters (k={int(best['k'])})"
)

plt.show()
# =====================================================
# 10. CLUSTER COUNTS AND PROFILES
# =====================================================

df["Cluster"] = final_labels

print("\nCluster Counts:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

if len(num_cols) > 0:

    print("\nCluster Profiles:")

    display(
        df.groupby("Cluster")[
            num_cols
        ].mean()
    )