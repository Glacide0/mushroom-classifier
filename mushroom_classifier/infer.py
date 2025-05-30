"""Inference module for mushroom classifier."""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from mushroom_classifier.train import MushroomClassifier

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class MushroomPredictor:
    """Class for making predictions with the trained model."""

    def __init__(self, config: DictConfig):
        """Initialize the predictor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            self.config.inference.device
            if torch.cuda.is_available() and "cuda" in self.config.inference.device
            else "cpu"
        )
        self.model = self._load_model()
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()

    def _load_class_mapping(self) -> Dict:
        """Load the class mapping from JSON file.

        Returns:
            Dictionary mapping class indices to class names
        """
        mapping_path = Path(self.config.paths.data.processed) / "class_mapping.json"
        if not mapping_path.exists():
            print(f"Warning: Class mapping file not found at {mapping_path}")
            return {}
        
        with open(mapping_path, 'r') as f:
            return json.load(f)

    def _load_model(self) -> Union[torch.nn.Module, ort.InferenceSession]:
        """Load the model based on the configuration.

        Returns:
            The loaded model or inference session
        """
        # Check if we should use ONNX
        use_onnx = getattr(self.config.inference, 'use_onnx', False)
        if use_onnx:
            if ort is None:
                raise ImportError(
                    "ONNX Runtime is not installed. "
                    "Please install it with `pip install onnxruntime`."
                )

            onnx_path = os.path.join(self.config.paths.models.exported, "model.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"ONNX model not found at {onnx_path}"
                )

            # Set up ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Create inference session
            onnx_session = ort.InferenceSession(
                onnx_path, sess_options=sess_options
            )

            return onnx_session

        # Otherwise, use PyTorch model
        checkpoint_path = os.path.join(self.config.paths.models.checkpoints, "best_model.ckpt")
        if not os.path.exists(checkpoint_path):
            # Try to find any checkpoint file
            checkpoint_dir = Path(self.config.paths.models.checkpoints)
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoint_files:
                checkpoint_path = str(checkpoint_files[0])
            else:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        # Load PyTorch model
        model = MushroomClassifier.load_from_checkpoint(checkpoint_path, config=self.config)
        model.eval()
        model.to(self.device)
        return model

    def preprocess_data(self, data_file: str) -> torch.Tensor:
        """Preprocess tabular data from a CSV file.

        Args:
            data_file: Path to the CSV file with mushroom features

        Returns:
            Preprocessed feature tensor
        """
        try:
            # Load the CSV file
            df = pd.read_csv(data_file)
            print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
            
            # Check if target column is present and remove it
            if df.columns[0] == 'class':
                df = df.drop(columns=['class'])
            
            # One-hot encode categorical features
            if getattr(self.config.feature_processing, "one_hot_encode", True):
                df = pd.get_dummies(df)
                
            # Convert to tensor
            features = torch.FloatTensor(df.values)
            
            print(f"Preprocessed features shape: {features.shape}")
            return features
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            raise

    def predict(self, data_file: str) -> Dict:
        """Make predictions for data in a CSV file.

        Args:
            data_file: Path to the CSV file

        Returns:
            Dictionary with prediction results
        """
        # Preprocess the data
        feature_tensor = self.preprocess_data(data_file)
        
        # Check if using ONNX
        use_onnx = getattr(self.config.inference, 'use_onnx', False)
        
        if not use_onnx:
            # Move to appropriate device
            feature_tensor = feature_tensor.to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(feature_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
        else:
            # For ONNX Runtime
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: feature_tensor.numpy()})
            probabilities = torch.nn.functional.softmax(torch.from_numpy(outputs[0]), dim=1)
        
        # Process results
        results = []
        for i in range(len(feature_tensor)):
            # Get top prediction for each sample
            probs = probabilities[i]
            confidence, predicted_idx = torch.max(probs, dim=0)
            predicted_class = predicted_idx.item()
            confidence_value = confidence.item()
            
            # Map to class name
            class_name = self.class_mapping.get(str(predicted_class), f"Class {predicted_class}")
            
            # Create result dictionary
            result = {
                "sample_id": i,
                "predicted_class": predicted_class,
                "class_name": class_name,
                "confidence": confidence_value,
                "probabilities": {cls_idx: probs[cls_idx].item() for cls_idx in range(len(probs))}
            }
            results.append(result)
        
        return {"predictions": results, "file_path": data_file}

    def batch_predict(self, data_files: List[str]) -> List[Dict]:
        """Make predictions for multiple data files.

        Args:
            data_files: List of paths to CSV files

        Returns:
            List of dictionaries with prediction results
        """
        return [self.predict(file) for file in data_files]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def infer(config: DictConfig) -> None:
    """Run inference with the trained model.

    Args:
        config: Hydra configuration
    """
    # Print the configuration
    print(OmegaConf.to_yaml(config))

    # Get the input path
    input_path = config.inference.input
    if not input_path:
        raise ValueError("Input path is required")

    # Initialize the predictor
    predictor = MushroomPredictor(config)

    # Make predictions
    if os.path.isdir(input_path):
        # Get all CSV files in the directory
        data_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".csv")
        ]
        results = predictor.batch_predict(data_files)
    else:
        # Single file
        results = predictor.predict(input_path)

    # Save the results
    if config.inference.output:
        output_path = config.inference.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        # Print the results
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    infer() 