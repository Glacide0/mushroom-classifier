#!/usr/bin/env python
"""Set up and start the Triton Inference Server."""

import os
import sys
import argparse
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mushroom_classifier.utils import setup_triton_server


def prepare_model_repository(model_repo_path: str, model_path: str, model_format: str = "onnx") -> str:
    """Prepare the model repository for Triton Inference Server.

    Args:
        model_repo_path: Path to the model repository
        model_path: Path to the model file
        model_format: Model format (onnx or tensorrt)

    Returns:
        Path to the prepared model repository
    """
    # Create model repository structure
    os.makedirs(model_repo_path, exist_ok=True)
    
    # Create mushroom_classifier model directory
    model_dir = os.path.join(model_repo_path, "mushroom_classifier")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create version directory
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy model file to version directory
    if model_format == "onnx":
        dst_path = os.path.join(version_dir, "model.onnx")
        platform = "onnxruntime_onnx"
    else:  # tensorrt
        dst_path = os.path.join(version_dir, "model.plan")
        platform = "tensorrt_plan"
    
    shutil.copy(model_path, dst_path)
    
    # Create config.pbtxt
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(f"""name: "mushroom_classifier"
platform: "{platform}"
max_batch_size: 8
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }}
]
""")
    
    return model_repo_path


def main() -> None:
    """Set up and start the Triton Inference Server."""
    parser = argparse.ArgumentParser(description="Set up Triton Inference Server")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/exported/model.onnx",
        help="Path to the model (ONNX or TensorRT)",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt"],
        help="Model format",
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="models/triton_model_repository",
        help="Path to the Triton model repository",
    )
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return

    # Prepare model repository
    model_repo = prepare_model_repository(args.model_repo, args.model_path, args.model_format)
    print(f"Model repository prepared at {model_repo}")

    # Start Triton server
    print(f"Starting Triton server with model repository at {model_repo}")
    setup_triton_server(model_repo)

    print("Triton server is running")
    print("Access the Triton API at http://localhost:8000/v2")
    print("Press Ctrl+C to stop the server")

    # Keep the script running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down Triton server")


if __name__ == "__main__":
    main() 