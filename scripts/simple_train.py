"""Standalone training script for mushroom classifier."""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset


class MushroomDataset(Dataset):
    """Dataset for mushroom classification from tabular data."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), self.labels[idx]


class MushroomClassifier(pl.LightningModule):
    """PyTorch Lightning module for mushroom classification."""

    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Create network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=5, 
            min_lr=0.00001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }


def main():
    """Run the training pipeline."""
    print("Starting training...")
    
    # Set the seed for reproducibility
    pl.seed_everything(42)
    
    # Configuration
    batch_size = 128
    num_workers = 4
    max_epochs = 100
    hidden_sizes = [256, 128, 64]
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    
    # Load and preprocess the data
    data_path = "data/raw/mushrooms.csv"
    
    # Try to find mushrooms.csv if it doesn't exist in the expected location
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Looking for mushrooms.csv in other locations...")
        
        # Try to find mushrooms.csv in the workspace
        for root, dirs, files in os.walk("."):
            if "mushrooms.csv" in files:
                # Found the file, copy it to the expected location
                import shutil
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                shutil.copy(os.path.join(root, "mushrooms.csv"), data_path)
                print(f"Copied mushrooms.csv to {data_path}")
                break
        else:
            print("Error: Could not find mushrooms.csv anywhere in the workspace.")
            return
    
    try:
        # Load the data
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # First column is the target (edible/poisonous)
        target_col = df.columns[0]
        
        # Map target values to numeric
        target_mapping = {label: idx for idx, label in enumerate(df[target_col].unique())}
        y = df[target_col].map(target_mapping).values
        
        # One-hot encode features
        X_df = df.drop(columns=[target_col])
        X_encoded = pd.get_dummies(X_df)
        X = X_encoded.values
        
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Number of features after one-hot encoding: {X.shape[1]}")
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset = MushroomDataset(X_train, y_train)
        val_dataset = MushroomDataset(X_val, y_val)
        test_dataset = MushroomDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        # Create the model
        print("Creating model...")
        model = MushroomClassifier(
            input_size=X.shape[1],
            hidden_sizes=hidden_sizes,
            num_classes=len(target_mapping),
            dropout_rate=0.3,
            learning_rate=0.001
        )
        
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath="models/checkpoints",
            filename="mushroom_classifier-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
        
        # Create logger
        logger = TensorBoardLogger(
            save_dir="logs/tensorboard",
            name="mushroom-classifier"
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=logger,
            gradient_clip_val=0.5
        )
        
        # Train the model
        print("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Test the model
        print("Testing model...")
        trainer.test(model, test_loader)
        
        print("Training completed!")
        print(f"Model checkpoints saved to models/checkpoints")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 