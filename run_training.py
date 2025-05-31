"""Script to run mushroom classifier training."""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the train function from the mushroom_classifier module
from mushroom_classifier.train import train

if __name__ == "__main__":
    # Run the training
    print("Starting mushroom classifier training...")
    train() 