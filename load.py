import os
import sys
import torch
from torch_geometric.loader import DataLoader

# Ensure the Material-ML-Model folder is in the Python path
sys.path.append(os.path.abspath("Material-ML-Model"))

# Import MaterialML from ML.py
from ML import MaterialML

# Define paths
dft_data_path = os.path.join("Material-ML-Model", "DFT_data")
model_path = os.path.join("Material-ML-Model", "models", "best_model.pth")

# Initialize MaterialML instance
ml = MaterialML(dft_data_path)

# Load trained model
ml.load_model(model_path)

# Prepare data
train_loader, val_loader = ml.prepare_data()

# Make predictions on validation set
ml.model.eval()
print("\nPrediction Examples:")
with torch.no_grad():
    for batch in val_loader:
        pred = ml.predict(batch)
        # Print predictions for each graph in the batch
        for i in range(len(pred)):
            print(f"\nGraph {i+1} in batch:")
            print(f"Predicted energy: {pred[i].item():.4f} eV")
            print(f"Actual energy: {batch.y[i].item():.4f} eV")
            print(f"Absolute error: {abs(pred[i].item() - batch.y[i].item()):.4f} eV")
        break  # Just show first batch
