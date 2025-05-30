#!/usr/bin/env python
"""Convert PyTorch model to ONNX format."""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mushroom_classifier.train import MushroomClassifier


@hydra.main(config_path="../configs", config_name="config")
def convert_to_onnx(config: DictConfig) -> None:
    """Convert PyTorch model to ONNX format.

    Args:
        config: Hydra configuration
    """
    print(OmegaConf.to_yaml(config))

    # Check if the checkpoint exists
    checkpoint_path = config.model.checkpoint.dirpath
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint directory {checkpoint_path} does not exist")
        return

    # Find the best checkpoint
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_path)
        if f.endswith(".ckpt") and "best" in f
    ]

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_path}")
        return

    best_checkpoint = os.path.join(checkpoint_path, checkpoint_files[0])
    print(f"Using checkpoint: {best_checkpoint}")

    # Load the model
    model = MushroomClassifier.load_from_checkpoint(best_checkpoint, config=config)
    model.eval()
    model.to("cpu")

    # Create a dummy input
    dummy_input = torch.randn(
        1, 3, config.preprocessing.image_size.height, config.preprocessing.image_size.width
    )

    # Create export directory if it doesn't exist
    export_dir = config.paths.models.exported
    os.makedirs(export_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(export_dir, "model.onnx")
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


if __name__ == "__main__":
    convert_to_onnx() 