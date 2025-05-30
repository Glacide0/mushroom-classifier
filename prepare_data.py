"""Prepare mushroom dataset for the classifier."""

import os
import pandas as pd
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print("Download successful.")
        return True
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return False

def main():
    # Create data directories if they don't exist
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(processed_dir / "train", exist_ok=True)
    os.makedirs(processed_dir / "val", exist_ok=True)
    os.makedirs(processed_dir / "test", exist_ok=True)
    
    # Define the dataset URL (UCI ML Repository)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    raw_data_path = raw_dir / "mushrooms.csv"
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(raw_data_path):
        success = download_file(url, raw_data_path)
        if not success:
            print("Failed to download data. Exiting.")
            return
    else:
        print(f"File already exists at {raw_data_path}. Skipping download.")
    
    # Define column names based on UCI data
    column_names = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", 
        "gill-attachment", "gill-spacing", "gill-size", "gill-color", 
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", 
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]
    
    # Process the dataset
    print("Processing dataset...")
    try:
        # Load the dataset without header
        df = pd.read_csv(raw_data_path, header=None)
        
        # Check for and remove duplicate header rows
        if df.iloc[0, 0] == 'class' or df.iloc[1, 0] == 'class':
            # Find all rows that contain 'class' in first column
            header_rows = df.index[df.iloc[:, 0] == 'class'].tolist()
            if header_rows:
                print(f"Found {len(header_rows)} header rows at indices: {header_rows}")
                # Drop all these rows
                df = df.drop(header_rows).reset_index(drop=True)
                print(f"Dropped header rows. New shape: {df.shape}")

        # Now set the column names
        df.columns = column_names
        
        # Display information about the dataset
        print(f"Dataset shape: {df.shape}")
        print("\nSample data:")
        print(df.head())
        
        # Save the formatted dataset
        df.to_csv(raw_data_path, index=False)
        print(f"\nSaved formatted dataset to {raw_data_path}")
        
        # Also create a class mapping file
        class_values = df['class'].unique()
        class_mapping = {i: val for i, val in enumerate(class_values)}
        class_mapping_df = pd.DataFrame.from_dict(class_mapping, orient='index')
        class_mapping_df.to_json(processed_dir / "class_mapping.json")
        print(f"Class mapping saved to {processed_dir / 'class_mapping.json'}")
        
        print("\nData preparation completed successfully!")
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main() 