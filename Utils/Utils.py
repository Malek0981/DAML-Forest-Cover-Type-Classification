import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

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
            print("Download = True : Downloading and saving the dataset as Parquet files...")
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