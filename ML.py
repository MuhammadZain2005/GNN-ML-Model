import torch
from torch_geometric.loader import DataLoader
from DFT_processor_2_Zain import DFTProcessor
from gnn_model import EnhancedGNNModel, GNNModel
import os
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt

class MaterialML:
    def __init__(self, dft_data_path, use_enhanced_model=True):
        """
        Initialize with path to DFT data
        
        Args:
            dft_data_path: Path to DFT calculation data
            use_enhanced_model: Whether to use the enhanced model with edge attributes
        """
        self.dft_data_path = dft_data_path
        self.graphs = None  # Will store processed graphs
        self.model = None
        self.use_enhanced_model = use_enhanced_model
        
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
    def prepare_data(self):
        """Process DFT data and prepare training set"""
        processor = DFTProcessor(self.dft_data_path)
        self.graphs = processor.process_directory()
        
        if not self.graphs:
            raise ValueError("No valid graphs were created. Check DFT data and processing steps.")
        
        # Split into training and validation sets (80-20 split)
        n_train = int(0.8 * len(self.graphs))
        train_graphs = self.graphs[:n_train]
        val_graphs = self.graphs[n_train:]
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        
        # Analyze the created graphs to verify PBC implementation
        self.analyze_graph_connectivity()
        
        return train_loader, val_loader
    
    def analyze_graph_connectivity(self):
        """Analyze graph connectivity to verify PBC implementation"""
        if not self.graphs:
            print("No graphs available for analysis")
            return
            
        # Sample a few graphs for analysis
        sample_size = min(5, len(self.graphs))
        sampled_graphs = self.graphs[:sample_size]
        
        print("\n=== Graph Connectivity Analysis ===")
        for i, graph in enumerate(sampled_graphs):
            n_nodes = graph.x.size(0)
            n_edges = graph.edge_index.size(1)
            avg_degree = n_edges / n_nodes
            
            # Count how many edges cross periodic boundaries
            if hasattr(graph, 'lattice'):
                lattice = graph.lattice.numpy()
                positions = graph.x[:, 1:4].numpy()  # Extract positions (x,y,z)
                
                edge_index = graph.edge_index.numpy()
                edge_attrs = graph.edge_attr.numpy() if hasattr(graph, 'edge_attr') else None
                
                # Calculate box dimensions
                box_dims = np.array([np.linalg.norm(vec) for vec in lattice])
                
                # Count edges that cross boundaries
                boundary_edges = 0
                for e in range(edge_index.shape[1]):
                    i, j = edge_index[0, e], edge_index[1, e]
                    pos_i, pos_j = positions[i], positions[j]
                    
                    # Direct distance
                    direct_dist = np.linalg.norm(pos_i - pos_j)
                    
                    # Minimum image distance using PBC
                    min_dist, _ = DFTProcessor.minimum_image_distance(pos_i, pos_j, lattice)
                    
                    # If minimum image distance is significantly different from direct distance,
                    # then this edge crosses a boundary
                    if abs(direct_dist - min_dist) > 0.1:  # Threshold for detecting boundary crossing
                        boundary_edges += 1
                
                print(f"Graph {i+1}: {n_nodes} atoms, {n_edges} edges, avg degree: {avg_degree:.2f}")
                print(f"Edges crossing periodic boundaries: {boundary_edges}")
                print(f"Box dimensions: {box_dims}")
                print()
# CHANGED PATIENCE FROM 20 --> 30 & LR FROM 0.0005 --> 0.0001
    def train(self, train_loader, val_loader, hidden_dim=128, lr=0.0001, epochs=200):
        """Train GNN model with updated parameters"""
        input_dim = train_loader.dataset[0].x.shape[1]  # Number of node features
        output_dim = 1  # Energy prediction (assumed to be energy per atom)
        
        # Choose model type based on initialization parameter
        if self.use_enhanced_model:
            self.model = EnhancedGNNModel(input_dim, hidden_dim, output_dim)
            model_path = os.path.join(self.models_dir, "best_enhanced_model.pth")
            print("Using EnhancedGNNModel with edge attributes for PBC")
        else:
            self.model = GNNModel(input_dim, hidden_dim, output_dim)
            model_path = os.path.join(self.models_dir, "best_model.pth")
            print("Using original GNNModel")
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out.squeeze(-1), batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = self.model(batch)
                    val_loss += criterion(out.squeeze(-1), batch.y).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(model_path)
                print(f"New best model saved with val loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break # make val loss lower dont early stop 
        
        self.load_model(model_path)
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plot_path = os.path.join(self.models_dir, 'training_plot.png')
        plt.savefig(plot_path)
        print(f"Training plot saved to {plot_path}")
    
    def predict(self, data=None):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        with torch.no_grad():
            return self.model(data)
    
    def build_graph_with_pbc(self, lattice_vectors, atom_positions, atom_types, forces, energy, cutoff=3.0):
        """
        Build a graph for POSCAR-only predictions using enhanced periodic boundary conditions.
        """
        # Create a temporary processor instance and call its build_graph function:
        processor = DFTProcessor(self.dft_data_path)
        graph = processor.build_graph(atom_positions, atom_types, forces, energy, lattice_vectors, cutoff)
        return graph

    def predict_from_poscar(self, poscar_files, cutoff=2.8): # CHANGED CUTOFF FROM 3 to 2.8 current percent difference is 23.0949%
        """
        Make predictions using only POSCAR files with enhanced PBC support.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        processor = DFTProcessor(self.dft_data_path)
        graphs = []
        for poscar_file in poscar_files:
            # Read structure data (lattice vectors are properly scaled)
            lattice_vectors, atom_positions, n_boron, n_carbon, total_atoms = processor.read_POSCAR(poscar_file)
            atom_types = ["B"] * n_boron + ["C"] * n_carbon
            
            # Use dummy forces and dummy energy (they are not used for prediction)
            dummy_forces = [[0.0, 0.0, 0.0] for _ in range(total_atoms)]
            dummy_energy = 0.0
            
            # Build graph using the enhanced PBC implementation
            graph = self.build_graph_with_pbc(np.array(lattice_vectors), np.array(atom_positions),
                                                atom_types, dummy_forces, dummy_energy, cutoff)
            if graph is not None:
                graphs.append(graph)
                n_edges = graph.edge_index.size(1)
                avg_degree = n_edges / total_atoms
                print(f"Created graph from {poscar_file} with {total_atoms} atoms ({n_boron} B, {n_carbon} C), {n_edges} edges, avg degree: {avg_degree:.2f}")
            else:
                print(f"Failed to create graph from {poscar_file}")
        
        if not graphs:
            raise ValueError("Failed to create any valid graphs from the POSCAR files")
        
        loader = DataLoader(graphs, batch_size=len(graphs))
        
        self.model.eval()
        predictions = []
        atoms_counts = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch)  # Model predicts energy per atom
                for i in range(len(out)):
                    energy_per_atom = out[i].item()
                    # If total energy is desired, multiply by number of atoms:
                    n_atoms = batch.n_atoms[i].item() if hasattr(batch, 'n_atoms') else graphs[i].n_atoms
                    total_energy = energy_per_atom * n_atoms
                    predictions.append((energy_per_atom, total_energy))
                    atoms_counts.append(n_atoms)
        
        return list(zip(predictions, atoms_counts))

    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
            
        # Determine the type of model to load
        if "enhanced" in path:
            use_enhanced = True
        else:
            use_enhanced = self.use_enhanced_model
            
        if self.model is None:
            input_dim = self.graphs[0].x.shape[1] if self.graphs else 7
            if use_enhanced:
                self.model = EnhancedGNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1)
            else:
                self.model = GNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1)
                
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    dft_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFT_data", "DFT_data")
    print(f"Using DFT data path: {dft_path}")
    
    try:
        ml = MaterialML(dft_path)
        train_loader, val_loader = ml.prepare_data()
        
        # Print training data statistics (energy per atom, system size, etc.)
        for batch in train_loader:
            energies_per_atom = batch.y
            n_atoms = batch.n_atoms
            total_energies = energies_per_atom * n_atoms
            print("\nTraining Data Statistics:")
            print(f"Energy per atom range: {energies_per_atom.min():.4f} to {energies_per_atom.max():.4f} eV/atom")
            print(f"Energy per atom mean: {energies_per_atom.mean():.4f} eV/atom")
            print(f"Energy per atom std: {energies_per_atom.std():.4f} eV/atom")
            print(f"Total energy range: {total_energies.min():.4f} to {total_energies.max():.4f} eV")
            print(f"Total energy mean: {total_energies.mean():.4f} eV")
            print(f"Total energy std: {total_energies.std():.4f} eV")
            print(f"System size range: {n_atoms.min().item()} to {n_atoms.max().item()} atoms")
            break
        
        ml.train(train_loader, val_loader)
        
        # Predictions on validation set
        ml.model.eval()
        print("\nPrediction Examples:")
        with torch.no_grad():
            for batch in val_loader:
                pred = ml.predict(batch)
                print("\nBatch Statistics:")
                print(f"Batch size: {len(pred)} systems")
                total_abs_error_per_atom = 0
                total_abs_error = 0
                for i in range(len(pred)):
                    n_atoms = batch.n_atoms[i]
                    pred_total = pred[i].item() * n_atoms
                    actual_total = batch.y[i].item() * n_atoms
                    error_per_atom = abs(pred[i].item() - batch.y[i].item())
                    error_total = abs(pred_total - actual_total)
                    total_abs_error_per_atom += error_per_atom
                    total_abs_error += error_total
                    print(f"\nGraph {i+1} in batch:")
                    print(f"Number of atoms: {n_atoms}")
                    print(f"Predicted energy per atom: {pred[i].item():.4f} eV/atom")
                    print(f"Actual energy per atom: {batch.y[i].item():.4f} eV/atom")
                    print(f"Predicted total energy: {pred_total:.4f} eV")
                    print(f"Actual total energy: {actual_total:.4f} eV")
                    print(f"Absolute error per atom: {error_per_atom:.4f} eV/atom")
                    print(f"Absolute total error: {error_total:.4f} eV")
                avg_error_per_atom = total_abs_error_per_atom / len(pred)
                avg_error_total = total_abs_error / len(pred)
                print(f"\nBatch Summary:")
                print(f"Average absolute error per atom: {avg_error_per_atom:.4f} eV/atom")
                print(f"Average absolute total error: {avg_error_total:.4f} eV")
                break
        
        # Test POSCAR-only predictions
        print("\nTesting prediction from POSCAR files:")
        model_path = os.path.join(ml.models_dir, "best_enhanced_model.pth")
        if os.path.exists(model_path):
            ml.load_model(model_path)
            unit_cell_poscar = "DFT_data/DFT_data/POSCAR/Unit_Super/POSCAR_unitcell.vasp"
            supercell_poscar = "DFT_data/DFT_data/POSCAR/Unit_Super/POSCAR_supercell.vasp"
            if os.path.exists(unit_cell_poscar) and os.path.exists(supercell_poscar):
                predictions = ml.predict_from_poscar([unit_cell_poscar, supercell_poscar])
                print("\nPOSCAR Prediction Results:")
                
                unit_prediction = predictions[0]
                unit_energy = unit_prediction[0][0] if isinstance(unit_prediction[0], tuple) else unit_prediction[0]
                unit_n_atoms = unit_prediction[1]
                
                super_prediction = predictions[1]
                super_energy = super_prediction[0][0] if isinstance(super_prediction[0], tuple) else super_prediction[0]
                super_n_atoms = super_prediction[1]
                
                print(f"Unit cell energy per atom: {unit_energy:.6f} eV/atom (atoms: {unit_n_atoms})")
                print(f"Supercell energy per atom: {super_energy:.6f} eV/atom (atoms: {super_n_atoms})")  
                        
                if abs(unit_energy) > 1e-6:
                    percent_diff = abs(super_energy - unit_energy) / abs(unit_energy) * 100
                    print(f"Percent difference: {percent_diff:.4f}%")
            else:
                print("POSCAR files not found. Update the paths to test this feature.")
    except Exception as e:
        print(f"Error: {str(e)}")


# Energy Parity Plot x= dft energies y= GNN energies (predicted) x=y line

# Try cutoff functions sinh tanh, 6.5 highest distance

# try to get better validation loss by running for more epochs and better early stopping approach.