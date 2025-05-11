import sys
import os
import torch
import numpy as np
from torch_geometric.data import Data, Batch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ML import MaterialML
from gnn_model import EquivariantGNNModel

def create_test_data(num_graphs=5, num_atoms=10):
    """
    Create synthetic test data for training and testing.
    """
    print("Creating test data...")
    
    graphs = []
    
    for g in range(num_graphs):
        # Features: [atomic_num, x, y, z, fx, fy, fz]
        x = torch.zeros((num_atoms, 7))
        
        # Random atomic numbers (5=B, 6=C)
        x[:, 0] = torch.tensor([5, 6] * (num_atoms // 2) + [5] * (num_atoms % 2))
        
        # Random positions
        positions = torch.rand((num_atoms, 3))
        x[:, 1:4] = positions
        
        # Create synthetic forces that depend on positions
        # These forces will be the negative gradient of a simple potential
        # We'll use a simple harmonic potential for demonstration
        center = torch.tensor([0.5, 0.5, 0.5])
        displacement = positions - center
        
        # Simple harmonic force: F = -k * displacement
        k = 0.5
        forces = -k * displacement
        x[:, 4:7] = forces
        
        # Energy proportional to sum of squared displacements
        energy = 0.5 * k * torch.sum(displacement ** 2) / num_atoms
        y = torch.tensor([energy.item()])
        
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
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.n_atoms = num_atoms
        
        graphs.append(data)
    
    return graphs

def test_energy_force_training():
    """
    Test the energy and force prediction training.
    """
    print("Testing energy and force prediction training...")
    
    # Create test data
    graphs = create_test_data(num_graphs=10, num_atoms=8)
    
    # Split into training and validation sets
    train_graphs = graphs[:8]
    val_graphs = graphs[8:]
    
    # Create a temporary directory for the models
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create the MaterialML instance
    ml = MaterialML(tmp_dir, use_enhanced_model=False, use_equivariant_model=True)
    
    # Override the graphs to use our synthetic test data
    ml.graphs = graphs
    
    # Create dataloaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
    
    # Train for a few epochs
    ml.train(train_loader, val_loader, hidden_dim=64, lr=0.01, epochs=10, w_forces=100.0)
    
    # Make predictions
    ml.model.eval()
    for data in val_graphs:
        # Add batch dimension for single graph
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        
        # Get predictions
        predictions = ml.predict(data)
        
        # Check that we have energy and forces
        assert 'energy' in predictions, "Missing energy in predictions"
        assert 'forces_direct' in predictions, "Missing direct forces in predictions"
        assert 'forces_autograd' in predictions, "Missing autograd forces in predictions"
        
        # Check shapes
        energy = predictions['energy']
        forces_direct = predictions['forces_direct']
        forces_autograd = predictions['forces_autograd']
        
        assert energy.shape == (1, 1), f"Energy shape should be (1, 1), got {energy.shape}"
        assert forces_direct.shape == (data.x.shape[0], 3), f"Forces shape should be {(data.x.shape[0], 3)}, got {forces_direct.shape}"
        assert forces_autograd.shape == (data.x.shape[0], 3), f"Autograd forces shape should be {(data.x.shape[0], 3)}, got {forces_autograd.shape}"
        
        # Print predictions
        print("\nPredictions:")
        print(f"Energy: {energy.item():.4f}")
        print(f"Target energy: {data.y.item():.4f}")
        
        # Print a couple of force predictions
        actual_forces = data.x[:, 4:7]
        print("\nForce predictions (sample):")
        for i in range(min(3, data.x.shape[0])):
            print(f"Atom {i+1}:")
            print(f"  Actual:   [{actual_forces[i, 0].item():.4f}, {actual_forces[i, 1].item():.4f}, {actual_forces[i, 2].item():.4f}]")
            print(f"  Direct:   [{forces_direct[i, 0].item():.4f}, {forces_direct[i, 1].item():.4f}, {forces_direct[i, 2].item():.4f}]")
            print(f"  Autograd: [{forces_autograd[i, 0].item():.4f}, {forces_autograd[i, 1].item():.4f}, {forces_autograd[i, 2].item():.4f}]")
        
        break  # Just test the first validation graph
    
    print("Energy and force prediction test completed successfully!")
    return True

if __name__ == "__main__":
    print("===== Testing Energy and Force Prediction =====")
    test_energy_force_training()
    print("\nAll tests completed!") 
