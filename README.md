# Mushroom Classifier

A machine learning classifier for identifying edible vs poisonous mushrooms from tabular data.

## Project Description

The goal of this project is to build a classifier that can identify whether mushrooms are edible or poisonous based on their features. The model is trained on the UCI Mushroom dataset, which contains categorical attributes describing various mushroom characteristics.

This project uses:
- PyTorch and PyTorch Lightning for model training
- Hydra for configuration management
- DVC for data versioning and management
- MLflow for experiment tracking
- ONNX and TensorRT for model optimization
- Triton Inference Server for model deployment

## Setup

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mushroom-classifier.git
cd mushroom-classifier
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR using conda
conda create -n mushroom-classifier python=3.8
conda activate mushroom-classifier
```

3. Install the package and dependencies using Poetry:
```bash
# Install Poetry if you don't have it
pip install poetry

# Install project dependencies
poetry install
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Data Management with DVC

The project uses DVC to manage and version the dataset. The data is stored in Google Drive by default.

1. Pull the data from the configured remote:
```bash
dvc pull
```

2. If you want to use a different storage provider, you can update the DVC configuration:
```bash
# For Google Drive
dvc remote add -d storage gdrive://your-gdrive-folder-id
dvc remote modify storage gdrive_acknowledge_abuse true

# For S3
dvc remote add -d s3storage s3://your-bucket-name
dvc remote modify s3storage endpointurl https://your-endpoint
dvc remote modify s3storage access_key_id your-access-key
dvc remote modify s3storage secret_access_key your-secret-key
```

## Training

Train the model using the default configuration:

```bash
python -m mushroom_classifier.train
```

Or with custom parameters:

```bash
python -m mushroom_classifier.train trainer.max_epochs=50 model.hidden_sizes=[256,128,64]
```

The training metrics will be logged to TensorBoard. You can view them with:

```bash
tensorboard --logdir logs/tensorboard
```

## Experiment Tracking with MLflow

Start the MLflow server to track experiments:

```bash
python scripts/start_mlflow_server.py --port 8080
```

Then open your browser and navigate to `http://127.0.0.1:8080` to view the MLflow UI.

Training metrics, parameters, and artifacts will be automatically logged to MLflow when you run the training script.

## Model Optimization

### Convert to ONNX

Convert the trained PyTorch model to ONNX format:

```bash
python scripts/convert_to_onnx.py --model_path models/checkpoints/best_model.ckpt --output_path models/exported/model.onnx
```

### Convert to TensorRT

Convert the ONNX model to TensorRT for faster inference:

```bash
python scripts/convert_to_tensorrt.py --onnx_path models/exported/model.onnx --tensorrt_path models/exported/model.engine --precision fp16
```

## Inference

Run inference using the trained model:

```bash
python -m mushroom_classifier.infer --input path/to/mushroom/features.csv --output predictions.json
```

To use the ONNX model for inference:

```bash
python -m mushroom_classifier.infer model.use_onnx=true --input path/to/mushroom/features.csv
```

## Deployment with Triton Inference Server

Set up and start the Triton Inference Server with your model:

```bash
python scripts/setup_triton_server.py --model_path models/exported/model.onnx --model_format onnx
```

This will:
1. Prepare the model repository with the correct structure
2. Configure the model for Triton
3. Start the Triton server

Once the server is running, you can send inference requests to `http://localhost:8000/v2/models/mushroom_classifier/infer` with an appropriate JSON payload.

Example inference request:

```bash
curl -X POST http://localhost:8000/v2/models/mushroom_classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3, 224, 224],
        "datatype": "FP32",
        "data": [...]
      }
    ]
  }'
```

## How We Will Check Your Work (Verification Steps)

1. **Setup and Installation**:
   - Clone the repository
   - Create a virtual environment
   - Run `poetry install`
   - Run `pre-commit install`

2. **Data Management**:
   - Run `dvc pull` to fetch the dataset
   - Verify data structure in the `data/` directory

3. **Training**:
   - Run `python -m mushroom_classifier.train`
   - Check TensorBoard logs in `logs/tensorboard/`
   - Check MLflow experiments at `http://127.0.0.1:8080`

4. **Model Export**:
   - Verify ONNX model was created in `models/exported/`
   - Run TensorRT conversion

5. **Inference**:
   - Run inference on test data
   - Start Triton server and test inference API

6. **Code Quality**:
   - Run `pre-commit run -a` to verify code quality

## Project Structure

```
mushroom_classifier/
├── configs/                # Hydra configuration files
│   ├── config.yaml         # Main configuration
│   ├── model.yaml          # Model configuration
│   ├── preprocessing.yaml  # Data preprocessing configuration
│   └── training.yaml       # Training configuration
├── data/                   # Data directory
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── logs/                   # Log files
│   ├── mlflow/             # MLflow logs
│   └── tensorboard/        # TensorBoard logs
├── models/                 # Model files
│   ├── checkpoints/        # Model checkpoints
│   └── exported/           # Exported models (ONNX, TensorRT)
├── mushroom_classifier/    # Main package
│   ├── __init__.py         # Package initialization
│   ├── data.py             # Data loading and preprocessing
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── infer.py            # Inference script
│   └── utils.py            # Utility functions
├── scripts/                # Utility scripts
│   ├── convert_to_onnx.py         # Convert model to ONNX
│   ├── convert_to_tensorrt.py     # Convert ONNX to TensorRT
│   ├── setup_triton_server.py     # Setup Triton server
│   └── start_mlflow_server.py     # Start MLflow server
├── .dvc/                   # DVC configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── pyproject.toml          # Poetry configuration
├── prepare_data.py         # Data preparation script
└── README.md               # Project documentation
```

## Dataset Information

The UCI Mushroom dataset contains features of 8,124 mushroom samples, each classified as either edible or poisonous. Features include:

- Cap shape, surface, color
- Bruises presence
- Odor
- Gill attachment, spacing, size, color
- Stalk shape and root
- Veil type and color
- Ring number and type
- Spore print color
- Population and habitat

For more information, see the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom). 