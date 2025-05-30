"""Data loading and preprocessing module for mushroom classifier."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from sklearn.model_selection import train_test_split


class MushroomDataset(Dataset):
    """Dataset for mushroom classification from tabular data."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        """Initialize the dataset.

        Args:
            features: Feature array
            labels: Labels array
            transform: Optional transforms to apply to the features
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (features, label)
        """
        feature = self.features[idx]
        label = self.labels[idx]

        # Convert to tensor
        feature_tensor = torch.FloatTensor(feature)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        return feature_tensor, label


class MushroomDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the mushroom dataset."""

    def __init__(self, config: DictConfig):
        """Initialize the data module.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.feature_names = None
        self.class_names = None
        self.num_features = None

    def prepare_data(self):
        """Prepare data if needed (e.g. download)."""
        # Import here to avoid import errors when first loading the config
        import dvc.api

        # Pull data from DVC if necessary
        try:
            data_path = self.config.paths.data.raw
            dvc.api.pull(data_path)
            print(f"Successfully pulled data from DVC: {data_path}")
        except Exception as e:
            print(f"Warning: Could not pull data from DVC: {e}")

    def setup(self, stage: str = None):
        """Set up the datasets.

        Args:
            stage: Optional stage parameter
        """
        # Load and process data
        data = self._load_and_preprocess_data()
        
        if data is None:
            raise ValueError("Failed to load data. Please check that the CSV file exists.")

        X, y, self.feature_names, self.class_names = data
        self.num_features = len(self.feature_names)
        
        # Update config with actual number of features
        if hasattr(self.config, "input_size"):
            self.config.input_size = self.num_features

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.config.data.val_ratio + self.config.data.test_ratio),
            random_state=self.config.seed
        )
        
        ratio = self.config.data.val_ratio / (self.config.data.val_ratio + self.config.data.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - ratio), random_state=self.config.seed
        )

        if stage == "fit" or stage is None:
            self.train_dataset = MushroomDataset(X_train, y_train)
            self.val_dataset = MushroomDataset(X_val, y_val)

        if stage == "test" or stage is None:
            self.test_dataset = MushroomDataset(X_test, y_test)

    def _load_and_preprocess_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Dict[int, str]]]:
        """Load and preprocess the mushroom dataset.

        Returns:
            Tuple of (features, labels, feature_names, class_names) or None if failed
        """
        # Path to the CSV file
        csv_path = os.path.join(self.config.paths.data.raw, "mushrooms.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            return None
            
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)
            print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
            
            # Assume first column is the target variable (edible/poisonous)
            # Adjust this based on your actual CSV structure
            target_col = df.columns[0]
            
            # Convert categorical target to numeric
            target_mapping = {label: idx for idx, label in enumerate(df[target_col].unique())}
            y = df[target_col].map(target_mapping).values
            
            # Create mapping from numeric target to class names
            class_names = {idx: label for label, idx in target_mapping.items()}
            print(f"Class mapping: {class_names}")
            
            # Process features (excluding target column)
            X_df = df.drop(columns=[target_col])
            feature_names = X_df.columns.tolist()
            
            # One-hot encode categorical features
            if getattr(self.config.feature_processing, "one_hot_encode", True):
                X_encoded = pd.get_dummies(X_df)
                feature_names = X_encoded.columns.tolist()
                X = X_encoded.values
            else:
                # Convert string columns to category codes
                for col in X_df.select_dtypes(include=['object']).columns:
                    X_df[col] = X_df[col].astype('category').cat.codes
                feature_names = X_df.columns.tolist()
                X = X_df.values
            
            print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
            return X, y, feature_names, class_names
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return None

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns:
            Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader.

        Returns:
            Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
        )

    def process_data(self) -> None:
        """Process the raw data and save intermediate results if needed."""
        print("Processing data...")
        data = self._load_and_preprocess_data()
        if data:
            X, y, feature_names, class_names = data
            print(f"Data processed successfully. Features shape: {X.shape}, Labels shape: {y.shape}")
            
            # Save class mapping to config directory
            import json
            class_mapping_path = os.path.join(self.config.paths.data.processed, "class_mapping.json")
            os.makedirs(os.path.dirname(class_mapping_path), exist_ok=True)
            
            with open(class_mapping_path, 'w') as f:
                json.dump({str(k): v for k, v in class_names.items()}, f)
            print(f"Class mapping saved to {class_mapping_path}")
            
            # Update the number of classes in config if needed
            num_classes = len(class_names)
            if hasattr(self.config, "num_classes"):
                self.config.num_classes = num_classes
                print(f"Updated num_classes to {num_classes}")
        else:
            print("Failed to process data.")


if __name__ == "__main__":
    # This allows running the data processing directly
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../configs", config_name="config")
    def process_data_main(config: DictConfig) -> None:
        """Process the data from the command line.

        Args:
            config: Hydra configuration
        """
        print(OmegaConf.to_yaml(config))
        data_module = MushroomDataModule(config)
        data_module.process_data()

    process_data_main() 