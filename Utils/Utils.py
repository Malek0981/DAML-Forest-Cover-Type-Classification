import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ACTIVE_FEATURES = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Soil_Type1',
    'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
    'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
    'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
    'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
    'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
    'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
    'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
    'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
    'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
    'Soil_Type40', 'Wilderness_Area2', 'Wilderness_Area3',
    'Wilderness_Area4'
]


def load_covertype_data(active_features: list = ACTIVE_FEATURES, flatten_target: bool = False, Download: bool = True):
    """
    Loads the Covertype dataset. If local Parquet files exist in DATA/raw/, 
    it loads them instantly. Otherwise, it downloads from UCI, saves them 
    as Parquet files if Download is True, and returns the filtered features and target.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "DATA", "raw")

    x_path = os.path.join(data_dir, "X.parquet")
    y_path = os.path.join(data_dir, "y.parquet")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading local dataset from '{data_dir}'")
        X_full = pd.read_parquet(x_path)
        y_full = pd.read_parquet(y_path)
    else:
        print(f"[Download] Local dataset not found in '{data_dir}'.")
        print("Fetching the Covertype dataset from the UCI Machine Learning Repository...")

        # Fetch from UCI repo
        covertype = fetch_ucirepo(id=31)
        X_full = covertype.data.features
        y_full = covertype.data.targets
        print("Dataset fetched successfully from UCI.")

        if Download:
            print(
                "Download = True : Downloading and saving the dataset as Parquet files...")
            # Ensure directory exists and save as Parquet
            os.makedirs(data_dir, exist_ok=True)
            X_full.to_parquet(x_path)
            y_full.to_parquet(y_path)

            print(f"Dataset securely saved to Parquet format in '{data_dir}'.")

    # Filter Check
    print("Filtering the dataset based on active features...")
    missing = [f for f in active_features if f not in X_full.columns]
    if missing:
        raise ValueError(f"Features not found in dataset: {missing}")

    X_filtered = X_full[active_features]

    # Formatting
    if flatten_target:
        print("Flattening the target variable...")
        y_filtered = y_full.values.ravel()
    else:
        y_filtered = y_full

    return X_filtered, y_filtered


# ---------------------------------------------------------
# DATA SPLITTING
# ---------------------------------------------------------

def split_data(
    X,
    y,
    method="stratified",
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Split a dataset into training, validation and test sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or pandas.DataFrame
        Target labels.
    method : {"random", "stratified"}
        Splitting strategy.
    train_size, val_size, test_size : float
        Fractions of the complete dataset.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test.
    """
    if method not in ("random", "stratified"):
        raise ValueError("method must be 'random' or 'stratified'")

    sizes = (train_size, val_size, test_size)
    if any(s <= 0 or s >= 1 for s in sizes):
        raise ValueError("Split fractions must be in (0, 1)")

    if not np.isclose(sum(sizes), 1.0):
        raise ValueError("Split fractions must sum to 1")

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    labels = np.asarray(y).ravel()
    stratify = labels if method == "stratified" else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    relative_val_size = val_size / (train_size + val_size)

    temp_labels = np.asarray(y_temp).ravel()
    stratify_temp = (
        temp_labels if method == "stratified" else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        stratify=stratify_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------
# PREPROCESSING CONFIGURATION
# ---------------------------------------------------------

@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration of the optional preprocessing steps."""

    scaling: bool = False
    pca: str = "none"  # "none", "numeric", "all"
    pca_components: int | float = 0.95
    rare_soil_threshold: float | None = None
    cyclic_aspect: bool = False


# ---------------------------------------------------------
# CYCLIC ASPECT TRANSFORMATION
# ---------------------------------------------------------

class AspectTransformer(BaseEstimator, TransformerMixin):
    """
    Replace Aspect (degrees) by Aspect_sin and Aspect_cos.

    Other features are kept unchanged.
    """

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        if "Aspect" not in X.columns:
            raise ValueError("Missing feature: Aspect")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()

        angle = np.deg2rad(X["Aspect"].to_numpy())

        X["Aspect_sin"] = np.sin(angle)
        X["Aspect_cos"] = np.cos(angle)

        return X.drop(columns="Aspect")

    def get_feature_names_out(self, input_features=None):
        features = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )

        return np.asarray(
            [f for f in features if f != "Aspect"]
            + ["Aspect_sin", "Aspect_cos"],
            dtype=object,
        )


# ---------------------------------------------------------
# RARE SOIL TYPE FILTERING
# ---------------------------------------------------------

class RareSoilFilter(BaseEstimator, TransformerMixin):
    """
    Remove Soil_Type columns with frequency below a threshold.

    Frequencies are estimated during fit, using training
    observations only.

    threshold : float
        Minimum required proportion of observations.
        For example, 0.001 corresponds to 0.1%.
    """

    def __init__(self, threshold=0.001):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        if not 0 <= self.threshold < 1:
            raise ValueError("threshold must be in [0, 1)")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        soil_cols = [
            col for col in X.columns
            if col.startswith("Soil_Type")
        ]

        frequencies = X[soil_cols].mean()

        self.removed_soils_ = frequencies[
            frequencies < self.threshold
        ].index.tolist()

        self.kept_columns_ = [
            col for col in X.columns
            if col not in self.removed_soils_
        ]

        return self

    def transform(self, X):
        return X.loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


# ---------------------------------------------------------
# FEATURE SELECTION HELPERS
# ---------------------------------------------------------

def _numeric_columns(X):
    """Select non-binary environmental features."""
    return [
        col for col in X.columns
        if not col.startswith(("Soil_Type", "Wilderness_Area"))
    ]


def _binary_columns(X):
    """Select one-hot encoded environmental categories."""
    return [
        col for col in X.columns
        if col.startswith(("Soil_Type", "Wilderness_Area"))
    ]


# ---------------------------------------------------------
# PREPROCESSOR BUILDER
# ---------------------------------------------------------

def build_preprocessor(config=None):
    """
    Construct an unfitted scikit-learn preprocessing pipeline.

    Processing order:
        1. Optional cyclic transformation of Aspect
        2. Optional rare soil filtering
        3. Scaling and optional PCA

    PCA modes:
        "none"    : No dimensionality reduction
        "numeric" : PCA on numerical features only
        "all"     : PCA on all remaining features

    Scaling is automatically applied before PCA.

    Parameters
    ----------
    config : PreprocessingConfig or None
        Configuration of the preprocessing pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted preprocessing pipeline.
    """
    if config is None:
        config = PreprocessingConfig()

    if not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be a PreprocessingConfig")

    if config.pca not in ("none", "numeric", "all"):
        raise ValueError(
            "pca must be 'none', 'numeric' or 'all'"
        )

    n = config.pca_components

    if isinstance(n, bool) or not (
        (isinstance(n, int) and n >= 1)
        or (isinstance(n, float) and 0 < n < 1)
    ):
        raise ValueError(
            "pca_components must be a positive integer "
            "or a float in (0, 1)"
        )

    steps = []

    # 1. Cyclic Aspect transformation
    if config.cyclic_aspect:
        steps.append(("aspect", AspectTransformer()))

    # 2. Rare Soil Type filtering
    if config.rare_soil_threshold is not None:
        steps.append((
            "rare_soils",
            RareSoilFilter(config.rare_soil_threshold),
        ))

    # 3. Scaling and PCA
    if config.pca == "all":
        # PCA over all features, including binary indicators.
        final_transformer = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
        ])

    else:
        if config.pca == "numeric":
            numeric_transformer = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n)),
            ])
        else:
            numeric_transformer = (
                StandardScaler()
                if config.scaling
                else "passthrough"
            )

        final_transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_transformer,
                    _numeric_columns,
                ),
                (
                    "binary",
                    "passthrough",
                    _binary_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    steps.append(("features", final_transformer))

    return Pipeline(steps)
