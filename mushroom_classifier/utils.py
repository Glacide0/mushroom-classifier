"""Utility functions for mushroom classifier."""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def get_git_commit_id() -> str:
    """Get the current git commit ID.

    Returns:
        The git commit ID or 'unknown' if not in a git repository
    """
    try:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], universal_newlines=True
        ).strip()
        return commit_id
    except subprocess.SubprocessError:
        return "unknown"


def convert_onnx_to_tensorrt(
    onnx_path: str,
    tensorrt_path: str,
    precision: str = "fp32",
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
) -> bool:
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model
        tensorrt_path: Path to save the TensorRT engine
        precision: Precision mode (fp32, fp16, or int8)
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes

    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError:
        print(
            "TensorRT is not installed or not found in the current environment."
            "Please install it following the instructions on the NVIDIA website."
        )
        return False

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(tensorrt_path), exist_ok=True)

    print(f"Converting {onnx_path} to TensorRT engine with precision {precision}")

    # Create logger
    logger = trt.Logger(trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # Set precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")
    else:
        print("Using FP32 precision")

    # Build engine
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
    opt_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])
    max_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)

    if engine is None:
        print("Failed to create TensorRT engine")
        return False

    # Serialize engine
    with open(tensorrt_path, "wb") as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {tensorrt_path}")
    return True


def plot_training_metrics(
    metrics: Dict[str, List[float]], save_path: Optional[str] = None
) -> None:
    """Plot training metrics.

    Args:
        metrics: Dictionary of metrics (e.g., {"train_loss": [...], "val_loss": [...]})
        save_path: Path to save the plot, if None, show the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    if "train_loss" in metrics:
        plt.plot(metrics["train_loss"], label="Training Loss")
    if "val_loss" in metrics:
        plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 1, 2)
    if "train_acc" in metrics:
        plt.plot(metrics["train_acc"], label="Training Accuracy")
    if "val_acc" in metrics:
        plt.plot(metrics["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def visualize_predictions(
    image_paths: List[str],
    predictions: List[Dict],
    class_names: Dict[str, str],
    num_samples: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Visualize model predictions.

    Args:
        image_paths: List of paths to images
        predictions: List of prediction dictionaries
        class_names: Dictionary mapping class indices to names
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization, if None, show the plot
    """
    # Choose random samples if there are more than num_samples
    num_images = min(len(image_paths), num_samples)
    indices = np.random.choice(len(image_paths), num_images, replace=False)

    plt.figure(figsize=(15, 4 * num_images))

    for i, idx in enumerate(indices):
        # Load image
        image = Image.open(image_paths[idx])

        # Get prediction
        pred = predictions[idx]
        pred_class = pred["predicted_class_name"]
        confidence = pred["confidence"]

        # Plot
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(image)
        plt.title(f"Prediction: {pred_class} (Confidence: {confidence:.2f})")
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def setup_mlflow_server(port: int = 8080) -> None:
    """Set up MLflow tracking server.

    Args:
        port: Port to run the server on
    """
    command = f"mlflow server --port {port}"
    print(f"Starting MLflow server with command: {command}")
    print(f"Access the MLflow UI at http://127.0.0.1:{port}")

    # Start the server as a background process
    try:
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("MLflow server started")
    except Exception as e:
        print(f"Error starting MLflow server: {e}")


def setup_triton_server(model_repository: str) -> None:
    """Set up Triton Inference Server.

    Args:
        model_repository: Path to the model repository
    """
    # Check if the model repository exists
    if not os.path.exists(model_repository):
        print(f"Model repository {model_repository} does not exist")
        return

    # Start Triton server as a background process
    command = f"tritonserver --model-repository {model_repository}"
    print(f"Starting Triton server with command: {command}")
    print("Access the Triton API at http://localhost:8000/v2")

    try:
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Triton server started")
    except Exception as e:
        print(f"Error starting Triton server: {e}")
        print("Make sure Triton Inference Server is installed and in your PATH")


def configure_dvc() -> None:
    """Configure DVC for the project.

    This function initializes DVC if it's not already initialized.
    """
    # Check if DVC is already initialized
    if os.path.exists(".dvc"):
        print("DVC is already initialized")
        return

    try:
        # Initialize DVC
        subprocess.check_call(["dvc", "init"])
        print("DVC initialized successfully")

        # Create data directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Add directories to DVC
        subprocess.check_call(["dvc", "add", "data/raw"])
        print("Added data/raw to DVC")

        # Ignore directories in git
        with open(".gitignore", "a") as f:
            f.write("\n/data\n/models\n")

        print("DVC setup completed successfully")
    except subprocess.SubprocessError as e:
        print(f"Error configuring DVC: {e}")
        print("Make sure DVC is installed and in your PATH")


if __name__ == "__main__":
    # This allows running utility functions directly
    import sys
    
    if len(sys.argv) > 1:
        # Simple CLI for utility functions
        command = sys.argv[1]
        
        if command == "setup_dvc":
            configure_dvc()
        elif command == "setup_mlflow":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
            setup_mlflow_server(port)
        elif command == "convert_to_tensorrt":
            if len(sys.argv) < 4:
                print("Usage: python utils.py convert_to_tensorrt <onnx_path> <tensorrt_path>")
                sys.exit(1)
            convert_onnx_to_tensorrt(sys.argv[2], sys.argv[3])
        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup_dvc, setup_mlflow, convert_to_tensorrt")
    else:
        print("No command specified")
        print("Available commands: setup_dvc, setup_mlflow, convert_to_tensorrt") 