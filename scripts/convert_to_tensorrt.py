#!/usr/bin/env python
"""Convert ONNX model to TensorRT engine."""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mushroom_classifier.utils import convert_onnx_to_tensorrt


def main() -> None:
    """Convert ONNX model to TensorRT engine."""
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="models/exported/model.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--tensorrt_path",
        type=str,
        default="models/exported/model.engine",
        help="Path to save the TensorRT engine",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode for the TensorRT engine",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=8, help="Maximum batch size"
    )
    args = parser.parse_args()

    # Check if the ONNX model exists
    if not os.path.exists(args.onnx_path):
        print(f"ONNX model not found at {args.onnx_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.tensorrt_path), exist_ok=True)

    # Convert ONNX to TensorRT
    success = convert_onnx_to_tensorrt(
        args.onnx_path, args.tensorrt_path, args.precision, args.max_batch_size
    )

    if success:
        print(
            f"Successfully converted {args.onnx_path} to TensorRT engine at {args.tensorrt_path}"
        )
    else:
        print(f"Failed to convert {args.onnx_path} to TensorRT")


if __name__ == "__main__":
    main() 