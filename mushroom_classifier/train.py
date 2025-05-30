"""Training module for mushroom classifier."""

import os
import sys
from typing import Dict, List, Union

import hydra
import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from mushroom_classifier.data import MushroomDataModule
from mushroom_classifier.model import build_model  # Import from Task 1 integration


class MushroomClassifier(pl.LightningModule):
    """PyTorch Lightning module for mushroom classification."""

    def __init__(self, config):
        """Initialize the model.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.save_hyperparameters(config)
        
        # Update these lines to access parameters directly from config root
        self.num_classes = config.num_classes
        self.input_size = config.input_size
        self.hidden_sizes = config.hidden_sizes
        self.dropout_rate = config.dropout_rate
        self.architecture = config.architecture
        
        # Build the model
        if self.architecture == "mlp":
            self._build_mlp()
        else:
            raise ValueError(f"Architecture {self.architecture} not supported")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _build_mlp(self):
        """Build a multi-layer perceptron model."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_sizes[-1], self.num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Model predictions
        """
        return self.model(x)

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Get parameters with gradients
        params = [p for p in self.parameters() if p.requires_grad]

        # Configure optimizer
        optimizer_name = self.hparams.optimizer.name.lower()
        lr = self.hparams.optimizer.lr
        weight_decay = self.hparams.optimizer.weight_decay

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

        # Configure scheduler
        scheduler_name = self.hparams.scheduler.name.lower()
        scheduler_config = {}

        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.scheduler.factor,
                patience=self.hparams.scheduler.patience,
                min_lr=self.hparams.scheduler.min_lr,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif scheduler_name == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.trainer.max_epochs,
                eta_min=self.hparams.scheduler.min_lr,
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.scheduler.patience,
                gamma=self.hparams.scheduler.factor,
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
        else:
            return {"optimizer": optimizer}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Dictionary with loss and other metrics
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return {"loss": loss, "train_acc": acc}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Dictionary with loss and other metrics
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Dictionary with loss and other metrics
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()

        # Calculate F1 score
        # This is a simplistic per-batch F1 calculation
        # In a real-world scenario, you'd accumulate predictions and compute at the end
        preds_cpu = preds.cpu()
        labels_cpu = labels.cpu()

        # Log metrics from Task 1
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {"test_loss": loss, "test_acc": acc}


def export_model_to_onnx(model: pl.LightningModule, config: DictConfig) -> None:
    """Export the model to ONNX format.

    Args:
        model: The trained model
        config: Configuration dictionary
    """
    model.to("cpu")
    model.eval()

    # Create a dummy input for tabular data with the right input size
    dummy_input = torch.randn(1, config.input_size)

    # Create the directory if it doesn't exist
    export_dir = config.paths.models.exported
    os.makedirs(export_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(export_dir, "model.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Model exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(config: DictConfig) -> None:
    """Train the model.

    Args:
        config: Hydra configuration
    """
    # Print the configuration
    print(OmegaConf.to_yaml(config))

    # Set random seed
    print(f"Seed set to {config.seed}")
    pl.seed_everything(config.seed)

    # Initialize the data module
    data_module = MushroomDataModule(config)
    
    # Setup data module to get actual feature count
    data_module.prepare_data()
    data_module.setup()
    
    # Update input_size in config with actual feature count
    if hasattr(data_module, "num_features"):
        config.input_size = data_module.num_features
        print(f"Updated input_size to match actual features: {config.input_size}")

    # Initialize the model
    model = MushroomClassifier(config)

    # Set up loggers
    tb_logs_dir = os.path.join(config.paths.logs.tensorboard, config.experiment.name)
    os.makedirs(tb_logs_dir, exist_ok=True)
    
    # Use TensorBoard logger as default
    logger = TensorBoardLogger(
        save_dir=tb_logs_dir,
        name=config.experiment.name,
    )
    
    # Only use MLflow if explicitly enabled and available
    use_mlflow = False  # Disable MLflow by default

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
    )

    early_stopping_callback = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        mode=config.early_stopping.mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        deterministic=config.trainer.deterministic,
        gradient_clip_val=config.trainer.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=logger,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

    # Export the model to ONNX
    export_model_to_onnx(model, config)
    
    # Log additional metrics used in Task 1
    # Add any specific metrics from Task 1 that should be logged


if __name__ == "__main__":
    train() 