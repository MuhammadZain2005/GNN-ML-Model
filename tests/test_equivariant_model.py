import torch
import sys
import os
import numpy as np
from torch_geometric.data import Data, Batch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnn_model import EquivariantGNNModel

def test_gradient_flow():
    """
    Test that gradients flow correctly through the positions in the model.
    """
    print("Testing gradient flow...")
    
    # Create a simple test graph
    num_atoms = 5
    
    # Features: [atomic_num, x, y, z, fx, fy, fz]
    x = torch.zeros((num_atoms, 7))
    x[:, 0] = torch.tensor([5, 5, 6, 6, 5])  # Atomic numbers (B=5, C=6)
    x[:, 1:4] = torch.rand((num_atoms, 3))  # Random positions
    x[:, 4:] = torch.zeros((num_atoms, 3))  # Zero forces
    
    # Create some edges (fully connected graph for simplicity)
    edge_index = []
    edge_attr = []
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edge_index.append([i, j])
                # Distance as edge attribute
                dist = torch.norm(x[i, 1:4] - x[j, 1:4])
                edge_attr.append([dist.item()])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr)
    
    # Create the data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Test with batch size 1
    batch = torch.zeros(num_atoms, dtype=torch.long)
    data.batch = batch
    
    # Initialize model
    input_dim = x.shape[1]
    model = EquivariantGNNModel(input_dim=input_dim, hidden_dim=64, output_dim=1)
    
    # Forward pass
    print("Running forward pass...")
    energy_pred, force_pred, pos = model(data)
    
    print(f"Energy prediction shape: {energy_pred.shape}")
    print(f"Force prediction shape: {force_pred.shape}")
    
    # Check that positions have gradients enabled
    print(f"Position requires grad: {pos.requires_grad}")
    
    # Test autograd forces
    print("Computing autograd forces...")
    forces = model.compute_autograd_forces(energy_pred, pos)
    
    print(f"Autograd forces shape: {forces.shape}")
    
    # Test that gradients flow through the whole model
    print("Testing backward pass...")
    loss = energy_pred.sum() + force_pred.sum()
    loss.backward()
    
    # If we get here without errors, gradients are flowing correctly
    print("Gradient flow test passed!")
    return True

def test_equivariance():
    """
    Test that the model predictions are equivariant to rotations.
    """
    print("\nTesting equivariance properties...")
    
    # Create a simple test graph
    num_atoms = 5
    
    # Features: [atomic_num, x, y, z, fx, fy, fz]
    x = torch.zeros((num_atoms, 7))
    x[:, 0] = torch.tensor([5, 5, 6, 6, 5])  # Atomic numbers (B=5, C=6)
    
    # Create specific positions in a known configuration
    positions = torch.tensor([
        [0.0, 0.0, 0.0],  # Origin
        [1.0, 0.0, 0.0],  # x-axis
        [0.0, 1.0, 0.0],  # y-axis
        [0.0, 0.0, 1.0],  # z-axis
        [1.0, 1.0, 1.0]   # Diagonal
    ])
    
    x[:, 1:4] = positions
    x[:, 4:] = torch.zeros((num_atoms, 3))  # Zero forces
    
    # Create some edges (fully connected graph for simplicity)
    edge_index = []
    edge_attr = []
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edge_index.append([i, j])
                # Distance as edge attribute
                dist = torch.norm(positions[i] - positions[j])
                edge_attr.append([dist.item()])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr)
    
    # Create the data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Test with batch size 1
    batch = torch.zeros(num_atoms, dtype=torch.long)
    data.batch = batch
    
    # Initialize model
    input_dim = x.shape[1]
    model = EquivariantGNNModel(input_dim=input_dim, hidden_dim=64, output_dim=1)
    
    # Get predictions for original positions
    model.eval()
    with torch.set_grad_enabled(True):
        energy_original, forces_original, _ = model(data)
        
    # Create a rotation matrix (90 degrees around z-axis)
    theta = torch.tensor(np.pi/2)
    rot_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Rotate positions
    rotated_positions = torch.matmul(positions, rot_matrix.T)
    
    # Create new data object with rotated positions
    x_rotated = x.clone()
    x_rotated[:, 1:4] = rotated_positions
    
    # Recreate edges with new distances
    edge_attr_rotated = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                dist = torch.norm(rotated_positions[i] - rotated_positions[j])
                edge_attr_rotated.append([dist.item()])
    
    edge_attr_rotated = torch.tensor(edge_attr_rotated)
    
    # Create the rotated data object
    data_rotated = Data(x=x_rotated, edge_index=edge_index, edge_attr=edge_attr_rotated)
    data_rotated.batch = batch
    
    # Get predictions for rotated positions
    with torch.set_grad_enabled(True):
        energy_rotated, forces_rotated, _ = model(data_rotated)
    
    # Energy should be invariant (same value)
    energy_diff = torch.abs(energy_original - energy_rotated).item()
    print(f"Energy difference after rotation: {energy_diff:.6f}")
    
    # Forces should be equivariant (rotate with the system)
    # Rotate the original forces for comparison
    forces_original_rotated = torch.matmul(forces_original, rot_matrix.T)
    
    # Calculate the error between rotated forces
    force_error = torch.mean(torch.abs(forces_original_rotated - forces_rotated)).item()
    print(f"Average force prediction error after rotation: {force_error:.6f}")
    
    # If errors are small, equivariance test is passed
    if energy_diff < 1e-4 and force_error < 1e-4:
        print("Equivariance test passed!")
        return True
    else:
        print("Equivariance test failed - errors too large")
        return False

def test_prediction_interface():
    """
    Test the prediction interface of the model.
    """
    print("\nTesting prediction interface...")
    
    # Create a simple test graph
    num_atoms = 5
    
    # Features: [atomic_num, x, y, z, fx, fy, fz]
    x = torch.zeros((num_atoms, 7))
    x[:, 0] = torch.tensor([5, 5, 6, 6, 5])  # Atomic numbers (B=5, C=6)
    x[:, 1:4] = torch.rand((num_atoms, 3))  # Random positions
    x[:, 4:] = torch.zeros((num_atoms, 3))  # Zero forces
    
    # Create some edges
    edge_index = []
    edge_attr = []
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edge_index.append([i, j])
                # Distance as edge attribute
                dist = torch.norm(x[i, 1:4] - x[j, 1:4])
                edge_attr.append([dist.item()])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr)
    
    # Create a batch of data
    data_list = [
        Data(x=x, edge_index=edge_index, edge_attr=edge_attr),
        Data(x=x, edge_index=edge_index, edge_attr=edge_attr)  # Same data for simplicity
    ]
    batch = Batch.from_data_list(data_list)
    
    # Initialize model
    input_dim = x.shape[1]
    model = EquivariantGNNModel(input_dim=input_dim, hidden_dim=64, output_dim=1)
    
    # Test prediction method
    predictions = model.predict(batch)
    
    # Check that all expected outputs are present
    expected_keys = ['energy', 'forces_direct', 'forces_autograd', 'positions']
    for key in expected_keys:
        assert key in predictions, f"Missing {key} in prediction output"
    
    # Check shapes
    assert predictions['energy'].shape[0] == 2, "Energy should have batch size 2"
    assert predictions['forces_direct'].shape[0] == num_atoms * 2, "Forces should have num_atoms * batch_size entries"
    
    print("Prediction interface test passed!")
    return True

if __name__ == "__main__":
    print("===== Testing Equivariant GNN Model =====")
    
    # Run tests
    test_gradient_flow()
    test_equivariance()
    test_prediction_interface()
    
    print("\nAll tests completed!") 