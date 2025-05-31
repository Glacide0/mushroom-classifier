"""Simplified script to convert ONNX model to TensorRT engine."""

import os
import sys
import argparse
import numpy as np
from typing import Tuple

def convert_onnx_to_tensorrt(
    onnx_path: str, 
    tensorrt_path: str, 
    precision: str = "fp32", 
    max_batch_size: int = 8,
    min_batch_size: int = 1,
    workspace_size: int = 1 << 30
) -> bool:
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model
        tensorrt_path: Path to save the TensorRT engine
        precision: Precision mode for the TensorRT engine
        max_batch_size: Maximum batch size
        min_batch_size: Minimum batch size
        workspace_size: Maximum workspace size in bytes

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Import TensorRT
        try:
            import tensorrt as trt
        except ImportError:
            print("TensorRT is not installed. Please install it first.")
            return False

        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Initialize builder
        builder = trt.Builder(TRT_LOGGER)
        
        # Create network
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        
        # Create parser
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
        else:
            print("Using FP32 precision")
        
        # Set dynamic batch size
        profile = builder.create_optimization_profile()
        for input_idx in range(network.num_inputs):
            input_tensor = network.get_input(input_idx)
            shape = input_tensor.shape
            profile.set_shape(
                input_tensor.name,
                (min_batch_size, *shape[1:]),
                (max_batch_size // 2, *shape[1:]),
                (max_batch_size, *shape[1:])
            )
        config.add_optimization_profile(profile)
        
        # Build engine
        print("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build TensorRT engine")
            return False
        
        # Serialize engine
        with open(tensorrt_path, "wb") as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {tensorrt_path}")
        return True
    
    except Exception as e:
        print(f"Error converting to TensorRT: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
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
        "--max_batch_size", 
        type=int, 
        default=8, 
        help="Maximum batch size"
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
        args.onnx_path,
        args.tensorrt_path,
        args.precision,
        args.max_batch_size
    )

    if success:
        print(f"Successfully converted {args.onnx_path} to TensorRT engine at {args.tensorrt_path}")
    else:
        print(f"Failed to convert {args.onnx_path} to TensorRT")


if __name__ == "__main__":
    main() 