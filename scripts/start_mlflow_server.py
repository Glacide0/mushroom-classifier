#!/usr/bin/env python
"""Start the MLflow server for experiment tracking."""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mushroom_classifier.utils import setup_mlflow_server


def main() -> None:
    """Start the MLflow server."""
    parser = argparse.ArgumentParser(description="Start MLflow server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the MLflow server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the MLflow server on",
    )
    args = parser.parse_args()

    print(f"Starting MLflow server at {args.host}:{args.port}")
    setup_mlflow_server(args.port)

    print("MLflow server is running")
    print(f"Access the MLflow UI at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")

    # Keep the script running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down MLflow server")


if __name__ == "__main__":
    main() 