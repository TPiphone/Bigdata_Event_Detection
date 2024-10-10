import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
import os
import glob

def load_data_from_gcs(bucket_folder_1, bucket_folder_2):
    """
    Load and combine CSV files from two GCS folders into a single DataFrame.
    
    Args:
        bucket_folder_1: Path to the first GCS folder (e.g., gs://space_things/NOT_STORM/).
        bucket_folder_2: Path to the second GCS folder (e.g., gs://space_things/STORM_LABELLED/).
        
    Returns:
        combined_df: Combined DataFrame containing data from both folders.
    """
    # Use gsutil to list all files in both folders
    not_storm_files = glob.glob(f'{bucket_folder_1}*.csv')
    storm_labelled_files = glob.glob(f'{bucket_folder_2}*.csv')

    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Load files from the first folder (NOT_STORM)
    for file_path in not_storm_files:
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Load files from the second folder (STORM_LABELLED)
    for file_path in storm_labelled_files:
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def create_lagged_features(data, n_lags=1, shift_steps=900):
    """
    Transform a time series into a supervised learning dataset.
    
    Args:
        data: Multivariate time series (DataFrame or 2D array)
        n_lags: Number of lagged time steps to include as features
        shift_steps: Number of steps to shift the target variable to predict ahead (900 for 3 hours)
    
    Returns:
        X: Feature matrix of lagged values
        Y: Target vector (binary labels 0 or 1) shifted by `shift_steps`
    """
    df = pd.DataFrame(data)
    
    # Create lagged features
    columns = [df.shift(i) for i in range(1, n_lags + 1)]
    columns.append(df.shift(-shift_steps))  # shift target column by shift_steps
    df_supervised = pd.concat(columns, axis=1)
    df_supervised.dropna(inplace=True)  # drop rows with NaN due to shifting
    
    # Split into inputs (X) and outputs (Y), assuming the last column is the target
    X = df_supervised.iloc[:, :-1].values  # all columns except the last one as features
    Y = (df_supervised.iloc[:, -1].values > 0.5).astype(int)  # binary 0/1 labels
    
    return X, Y

def main(args):
    # Load the data from two GCS folders
    data = load_data_from_gcs(args.not_storm_path, args.storm_labelled_path)
    
    # Preprocess data (e.g., creating lagged features)
    X, Y = create_lagged_features(data.values)
    
    # Train-test split
    test_size = 0.3
    split_idx = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Scaling the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Define and train the model
    svc = SVC(C=args.C, kernel=args.kernel, gamma=args.gamma)
    svc.fit(X_train_scaled, Y_train)

    # Evaluate the model
    Y_pred = svc.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(classification_report(Y_test, Y_pred))
    print(f"Accuracy: {accuracy:.4f}")

    # Save the model
    joblib.dump(svc, args.model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_storm_path', type=str, required=True, help='GCS path to the NOT_STORM folder')
    parser.add_argument('--storm_labelled_path', type=str, required=True, help='GCS path to the STORM_LABELLED folder')
    parser.add_argument('--model_output', type=str, default='gs://your-bucket/output/model.joblib')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--gamma', type=str, default='scale')
    args = parser.parse_args()
    
    main(args)
