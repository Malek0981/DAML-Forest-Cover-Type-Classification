# DAML-Forest-Cover-Type-Classification

Based on the Dataset : https://archive.ics.uci.edu/dataset/31/covertype
(Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N.)

The goal here is to predict forest cover type from cartographic variables only. This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.  These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

We will :
- Explore the data to look for null/incoherent values
- Build a simple model to have first results
- Build more complex models
- Explore the data again to see if we can get rid of some noisy features
- Adapt our models
- Benchmark them
- Analyze the results

# To use this code :

## Download uv : 

#### On macOS and Linux
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
```


#### On Windows
```bash 
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### With pip
```bash
pip install uv
```

#### With Homebrew.
```bash
brew install uv
```


## Install dependencies
In your terminal, run  :
```bash
uv sync
```


## Data Loading and Preprocessing

All data loading and preprocessing utilities are implemented in `Utils/Utils.py`.

The preprocessing pipeline is configurable and supports:

- Random or stratified train/validation/test splitting.
- Standardization of numerical features.
- Cyclic encoding of `Aspect` using sine and cosine.
- Optional removal of rare soil type features.
- PCA on numerical features only or on all features.

Preprocessing transformations are fitted exclusively on the training set to avoid data leakage.

### 1. Load the dataset

```python
from Utils.Utils import (
    load_covertype_data,
    split_data,
    PreprocessingConfig,
    build_preprocessor,
)

X, y = load_covertype_data()
```

The function returns the feature matrix `X` and target labels `y`. By default, all 54 features are loaded.

### 2. Split the dataset

```python
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y,
    method="stratified",
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
)
```

Two splitting methods are available:

- `method="random"`: randomly distributes observations.
- `method="stratified"` (default): preserves approximately the original class proportions across the three subsets.

The default split is 70% training, 15% validation and 15% testing.

### 3. Configure preprocessing

Preprocessing steps can be selected using `PreprocessingConfig`:

```python
config = PreprocessingConfig(
    scaling=True,
    pca="numeric",
    pca_components=0.95,
    rare_soil_threshold=None,
    cyclic_aspect=True,
)
```

| Parameter | Default | Description |
|---|---|---|
| `scaling` | `False` | Standardizes numerical features using `StandardScaler`. |
| `pca` | `"none"` | `"none"`, `"numeric"` (numerical features only), or `"all"` (all features). |
| `pca_components` | `0.95` | Positive integer for the number of components, or float between 0 and 1 for the fraction of variance to preserve. |
| `rare_soil_threshold` | `None` | Removes soil indicators whose training-set frequency is below the specified threshold (e.g. `0.001` for 0.1%). |
| `cyclic_aspect` | `False` | Replaces `Aspect` with `Aspect_sin` and `Aspect_cos`. |

**Note:** Scaling is automatically applied before PCA, regardless of the `scaling` parameter. For numerical PCA, binary indicators remain unchanged. For global PCA, all remaining features, including binary indicators, are standardized.

### 4. Apply preprocessing and retrieve processed data

```python
preprocessor = build_preprocessor(config)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to validation and test data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("Original shape:", X_train.shape)
print("Processed shape:", X_train_processed.shape)
```

The processed feature matrices are returned as NumPy arrays, ready to be used by classification algorithms.

The target labels (`y_train`, `y_val`, `y_test`) are not modified.

### 5. Use preprocessing directly with a classifier

Alternatively, the preprocessor can be integrated into a scikit-learn `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ("preprocessing", build_preprocessor(config)),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train.squeeze())

y_pred = model.predict(X_val)
```

This approach automatically fits preprocessing on the training set and applies the same transformations during prediction.

**Important:** Use the same data split and random seed when comparing classifiers or preprocessing configurations to ensure reproducible results.