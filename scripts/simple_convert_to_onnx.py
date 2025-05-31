"""Simplified script to convert PyTorch model to ONNX format."""

import os
import sys
import torch
import glob
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model class from the standalone script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_train import MushroomClassifier


def main():
    """Convert PyTorch model to ONNX format."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/exported/model.onnx",
        help="Path to save the ONNX model",
    )
    args = parser.parse_args()

    # Check if the checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"Checkpoint directory {args.checkpoint_dir} does not exist")
        return

    # Find the best checkpoint based on validation loss
    checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    # Sort checkpoints by validation loss (if in filename) or use last.ckpt
    best_checkpoint = None
    for file in checkpoint_files:
        if "last" in os.path.basename(file):
            best_checkpoint = file
            break
    
    # If no last.ckpt, use the first checkpoint with lowest val_loss
    if best_checkpoint is None:
        val_loss_checkpoints = [f for f in checkpoint_files if "val_loss" in os.path.basename(f)]
        if val_loss_checkpoints:
            # Sort by val_loss value in filename (lower is better)
            val_loss_checkpoints.sort(key=lambda x: float(os.path.basename(x).split("val_loss=")[1].split(".ckpt")[0]))
            best_checkpoint = val_loss_checkpoints[0]
        else:
            # Just use the first checkpoint
            best_checkpoint = checkpoint_files[0]

    print(f"Using checkpoint: {best_checkpoint}")

    # Load the checkpoint to get hyperparameters
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    
    # Extract hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})
    
    # If hyperparameters are missing, use some defaults
    input_size = hparams.get("input_size", 117)  # Default based on the mushroom dataset
    hidden_sizes = hparams.get("hidden_sizes", [256, 128, 64])
    num_classes = hparams.get("num_classes", 2)
    dropout_rate = hparams.get("dropout_rate", 0.3)
    learning_rate = hparams.get("learning_rate", 0.001)
    
    # Create a model with the same architecture
    model = MushroomClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Load the weights
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Create a dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Create export directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    print(f"Model exported to ONNX: {args.output_path}")
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model successfully validated!")
    except ImportError:
        print("ONNX package not installed, skipping validation.")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")


if __name__ == "__main__":
    main() 