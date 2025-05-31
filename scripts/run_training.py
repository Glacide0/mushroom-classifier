"""Script to run mushroom classifier training directly without DVC dependency."""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataclasses import dataclass
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model and training components
from mushroom_classifier.train import MushroomClassifier
from mushroom_classifier.data import MushroomDataModule

class SimpleConfig:
    """Simple configuration class to mimic DictConfig."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def main():
    """Run the training pipeline."""
    print("Starting training...")
    
    # Set the seed for reproducibility
    seed = 42
    pl.seed_everything(seed)
    
    # Create a config structure similar to what Hydra would provide
    paths = SimpleConfig(
        data=SimpleConfig(
            raw="data/raw",
            processed="data/processed",
            train="data/processed/train",
            val="data/processed/val",
            test="data/processed/test"
        ),
        models=SimpleConfig(
            checkpoints="models/checkpoints",
            exported="models/exported"
        ),
        logs=SimpleConfig(
            tensorboard="logs/tensorboard"
        )
    )
    
    feature_processing = SimpleConfig(one_hot_encode=True)
    
    data_config = SimpleConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    dataloader_config = SimpleConfig(
        batch_size=128,
        num_workers=4,
        pin_memory=True
    )
    
    # Create the main config
    config = SimpleConfig(
        paths=paths,
        feature_processing=feature_processing,
        data=data_config,
        dataloader=dataloader_config,
        seed=seed,
        num_classes=2,
        input_size=117,  # Will be updated dynamically
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.3,
        architecture="mlp"
    )
    
    # Create the data module with the config
    data_module = MushroomDataModule(config)
    
    # Check if the data file exists
    raw_data_path = os.path.join(config.paths.data.raw, "mushrooms.csv")
    if not os.path.exists(raw_data_path):
        print(f"Error: Data file not found at {raw_data_path}")
        print("Current working directory:", os.getcwd())
        print("Looking for mushrooms.csv in other locations...")
        
        # Try to find mushrooms.csv in the workspace
        for root, dirs, files in os.walk("."):
            if "mushrooms.csv" in files:
                # Found the file, copy it to the expected location
                import shutil
                os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
                shutil.copy(os.path.join(root, "mushrooms.csv"), raw_data_path)
                print(f"Copied mushrooms.csv to {raw_data_path}")
                break
        else:
            print("Error: Could not find mushrooms.csv anywhere in the workspace.")
            return
    
    # Prepare the data
    print("Preparing data...")
    try:
        # Skip DVC pull by monkeypatching the prepare_data method
        original_prepare_data = data_module.prepare_data
        data_module.prepare_data = lambda: None
        
        # Setup the data module
        data_module.setup()
        
        # Get the input size after one-hot encoding
        input_size = len(data_module.feature_names)
        print(f"Input size after one-hot encoding: {input_size}")
        
        # Create optimizer and scheduler configs
        optimizer = SimpleConfig(
            name="adam",
            lr=0.001,
            weight_decay=0.0001
        )
        
        scheduler = SimpleConfig(
            name="reduce_on_plateau",
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Update the config with the correct input size
        config.input_size = input_size
        
        # Create model config for initialization
        model_config = SimpleConfig(
            num_classes=config.num_classes,
            input_size=input_size,
            hidden_sizes=config.hidden_sizes,
            dropout_rate=config.dropout_rate,
            architecture=config.architecture,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Create the model
        print("Creating model...")
        model = MushroomClassifier(model_config)
        
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.paths.models.checkpoints,
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
            save_dir=config.paths.logs.tensorboard,
            name="mushroom-classifier"
        )
        
        # Create trainer config
        trainer_config = SimpleConfig(
            max_epochs=100,
            accelerator="auto",
            devices=1,
            precision=32,
            deterministic=True,
            gradient_clip_val=0.5
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=trainer_config.max_epochs,
            accelerator=trainer_config.accelerator,
            devices=trainer_config.devices,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=logger,
            gradient_clip_val=trainer_config.gradient_clip_val
        )
        
        # Train the model
        print("Starting training...")
        trainer.fit(model, data_module)
        
        # Test the model
        print("Testing model...")
        trainer.test(model, data_module)
        
        print("Training completed!")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 