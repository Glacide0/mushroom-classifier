"""Model module for the mushroom classifier."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

def build_model(config: DictConfig):
    """Build a multi-layer perceptron model for mushroom classification.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        The MLP model instance
    """
    input_size = config.input_size  # Should be actual feature count after one-hot encoding
    num_classes = config.num_classes  # Should be 2 for edible/poisonous
    hidden_sizes = config.hidden_sizes
    dropout_rate = config.dropout_rate
    
    # Create layers list
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    
    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    # Create sequential model
    model = nn.Sequential(*layers)
    
    return model

def preprocess_features(features, config):
    """Preprocess feature data.
    
    Args:
        features: Feature data (already as tensor or numpy array)
        config: The configuration dictionary
        
    Returns:
        Preprocessed tensor
    """
    # Convert to tensor if not already
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    
    # Add batch dimension if not present
    if len(features.shape) == 1:
        features = features.unsqueeze(0)
        
    return features

def predict(model, features, config):
    """Run inference using the model.
    
    Args:
        model: The trained model
        features: Preprocessed feature tensor
        config: The configuration dictionary
        
    Returns:
        Dictionary with prediction results
    """
    device = torch.device(config.inference.device if torch.cuda.is_available() and "cuda" in config.inference.device else "cpu")
    model = model.to(device)
    features = features.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(features)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # For batch processing
        if probabilities.shape[0] > 1:
            # Get predictions for each sample
            confidences, class_indices = torch.max(probabilities, dim=1)
            
            results = []
            for i, (confidence, class_idx) in enumerate(zip(confidences, class_indices)):
                # Get class mapping from config if available
                class_name = "Unknown"
                if hasattr(config, "class_mapping") and str(class_idx.item()) in config.class_mapping:
                    class_name = config.class_mapping[str(class_idx.item())]
                
                results.append({
                    "sample_id": i,
                    "class_id": class_idx.item(),
                    "class_name": class_name,
                    "confidence": confidence.item(),
                    "probabilities": {str(j): prob.item() for j, prob in enumerate(probabilities[i])}
                })
            return results
            
        # For single sample
        else:
            confidence, class_idx = torch.max(probabilities[0], dim=0)
            
            # Get class mapping from config if available
            class_name = "Unknown"
            if hasattr(config, "class_mapping") and str(class_idx.item()) in config.class_mapping:
                class_name = config.class_mapping[str(class_idx.item())]
            
            return {
                "class_id": class_idx.item(),
                "class_name": class_name,
                "confidence": confidence.item(),
                "probabilities": {str(i): prob.item() for i, prob in enumerate(probabilities[0])}
            } 