"""Mushroom classifier package."""

__version__ = "0.1.0"

# Package imports for easier access
from mushroom_classifier.data import MushroomDataModule, MushroomDataset
from mushroom_classifier.train import MushroomClassifier, export_model_to_onnx
from mushroom_classifier.infer import MushroomPredictor
from mushroom_classifier.model import build_model, preprocess_features, predict 