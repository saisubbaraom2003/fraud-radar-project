# scripts/preprocess.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the data by splitting, scaling, and handling class imbalance with SMOTE.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        A tuple containing the preprocessed data splits (X_train, X_test, y_train, y_test)
        and the fitted StandardScaler instance.
    """
    # Step 1: Create scaled 'Amount' and 'Time' columns.
    # It's good practice to use separate temporary scalers here
    # if you only want these two columns scaled initially, or
    # if you plan to scale all features with one global scaler later.
    # For simplicity and to align with the 30-feature expectation,
    # we'll create the scaled columns, then use ONE main_scaler on all features.

    # Use temporary scalers just to get the scaled values for these two specific columns
    amount_scaler_temp = StandardScaler()
    df['scaled_amount'] = amount_scaler_temp.fit_transform(df['Amount'].values.reshape(-1, 1))

    time_scaler_temp = StandardScaler()
    df['scaled_time'] = time_scaler_temp.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop the original 'Time' and 'Amount' columns, keeping the new scaled ones
    df.drop(['Time', 'Amount'], axis=1, inplace=True) 
    
    # Reorder columns to have 'Class' at the end (good practice, optional for functionality)
    df = df[[col for col in df if col != 'Class'] + ['Class']]

    # Step 2: Separate features (X) and target (y)
    # X will now contain V1-V28, scaled_amount, and scaled_time (total 30 features)
    X = df.drop('Class', axis=1) 
    y = df['Class']

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 4: Initialize and fit the MAIN StandardScaler on ALL training features.
    # This 'scaler' will be returned and saved, and it must be fitted on 30 features.
    main_scaler = StandardScaler()
    X_train_scaled = main_scaler.fit_transform(X_train) # FIT ON ALL 30 FEATURES
    X_test_scaled = main_scaler.transform(X_test)      # TRANSFORM TEST SET

    # Convert back to DataFrame to maintain column names and structure,
    # which is often helpful for subsequent steps like SMOTE or model training.
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Step 5: Apply SMOTE to handle imbalanced data on the SCALED training data
    minority_count = np.sum(y_train == 1) # y_train is the original count before SMOTE
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
    # Ensure k_neighbors is at least 1 if minority_count is 1 (SMOTE needs neighbors)
    if k_neighbors == 0 and minority_count > 0: k_neighbors = 1 
    elif minority_count == 0: k_neighbors = 0 # Handle case with no minority samples

    print(f"Applying SMOTE with k_neighbors={k_neighbors} (minority samples in train set: {minority_count})")
    
    # Only apply SMOTE if there are minority samples and k_neighbors is valid
    if minority_count > 0 and k_neighbors > 0:
        smote = SMOTE(sampling_strategy='minority', k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled_df, y_train) # Use X_train_scaled_df
    else:
        # If no SMOTE, use the original scaled training data
        print("Skipping SMOTE due to insufficient minority samples or k_neighbors constraint.")
        X_resampled, y_resampled = X_train_scaled_df, y_train


    # Step 6: Return the preprocessed data splits and the main fitted scaler
    return (X_resampled, X_test_scaled_df, y_resampled, y_test), main_scaler