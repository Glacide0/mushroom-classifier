"""Simple inference script for mushroom classifier."""

import os
import sys
import argparse
import numpy as np
import pandas as pd

def run_inference_onnx(model_path, input_data):
    """Run inference using ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        input_data: Input data as numpy array
    
    Returns:
        Prediction results
    """
    try:
        import onnxruntime as ort
        
        # Create ONNX inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options)
        
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        results = session.run([output_name], {input_name: input_data})
        return results[0]
    
    except ImportError:
        print("ONNX Runtime is not installed. Please install it with: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"Error running ONNX inference: {e}")
        return None


def main():
    """Run inference on a test dataset."""
    parser = argparse.ArgumentParser(description="Run inference using ONNX model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/exported/model.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/mushrooms.csv",
        help="Path to the test data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to run inference on",
    )
    args = parser.parse_args()
    
    # Check if the model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    
    # Check if the data exists
    if not os.path.exists(args.data_path):
        print(f"Data not found at {args.data_path}")
        return
    
    try:
        # Load the data
        df = pd.read_csv(args.data_path)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # First column is the target (edible/poisonous)
        target_col = df.columns[0]
        
        # Map target values to numeric and back for display
        target_mapping = {label: idx for idx, label in enumerate(df[target_col].unique())}
        reverse_mapping = {idx: label for label, idx in target_mapping.items()}
        y = df[target_col].map(target_mapping).values
        
        # One-hot encode features
        X_df = df.drop(columns=[target_col])
        X_encoded = pd.get_dummies(X_df)
        X = X_encoded.values
        
        # Select random samples
        np.random.seed(42)
        sample_indices = np.random.choice(len(X), min(args.num_samples, len(X)), replace=False)
        X_samples = X[sample_indices].astype(np.float32)
        y_samples = y[sample_indices]
        
        # Run inference
        print(f"Running inference on {len(X_samples)} samples...")
        predictions = run_inference_onnx(args.model_path, X_samples)
        
        if predictions is not None:
            # Get predicted classes
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Print results
            print("\nResults:")
            print("--------")
            for i, (pred, true) in enumerate(zip(predicted_classes, y_samples)):
                print(f"Sample {i+1}:")
                print(f"  True label: {reverse_mapping[true]}")
                print(f"  Predicted label: {reverse_mapping[pred]}")
                print(f"  Correct: {'Yes' if pred == true else 'No'}")
                print("--------")
            
            # Print overall accuracy
            accuracy = np.mean(predicted_classes == y_samples)
            print(f"Accuracy on samples: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 