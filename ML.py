import torch
from torch_geometric.loader import DataLoader
from DFT_processor_2_Zain import DFTProcessor
from gnn_model import GNNModel
import os

class MaterialML:
    def __init__(self, dft_data_path):
        """Initialize with path to DFT data"""
        self.dft_data_path = dft_data_path
        self.graphs = None  # Will store processed graphs
        self.model = None
        
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
        
        return train_loader, val_loader

    def train(self, train_loader, val_loader, hidden_dim=128, lr=0.0005, epochs=200):
        """Train GNN model with updated parameters"""
        input_dim = train_loader.dataset[0].x.shape[1]  # Number of node features
        output_dim = 1  # Energy prediction
        
        self.model = GNNModel(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_path = os.path.join(self.models_dir, "best_model.pth")
        
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
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = self.model(batch)
                    val_loss += criterion(out.squeeze(-1), batch.y).item()
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(best_model_path)
                print(f"New best model saved with val loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.load_model(best_model_path)
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    def predict(self, data=None):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        
        with torch.no_grad():
            return self.model(data)
    
    def predict_from_poscar(self, poscar_files):
        """Make predictions using only POSCAR files without requiring OUTCAR data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Initialize processor
        processor = DFTProcessor(self.dft_data_path)
        
        # Create graphs from POSCAR files
        graphs = []
        for poscar_file in poscar_files:
            # Read structure data
            lattice_vectors, atom_positions, n_boron, n_carbon, total_atoms = processor.read_POSCAR(poscar_file)
            
            # Create atom types list
            atom_types = ["B"] * n_boron + ["C"] * n_carbon
            
            # For prediction, we don't have forces or energy, so use dummy values
            dummy_forces = [[0.0, 0.0, 0.0] for _ in range(total_atoms)]
            dummy_energy = 0.0  # This will be replaced by the prediction
            
            # Build graph
            graph = processor.build_graph(atom_positions, atom_types, dummy_forces, dummy_energy)
            
            if graph is not None:
                graphs.append(graph)
                print(f"Created graph from {poscar_file} with {total_atoms} atoms ({n_boron} B, {n_carbon} C)")
            else:
                print(f"Failed to create graph from {poscar_file}")
        
        if not graphs:
            raise ValueError("Failed to create any valid graphs from the POSCAR files")
        
        # Create a batch from the graphs
        loader = DataLoader(graphs, batch_size=len(graphs))
        
        # Make predictions
        self.model.eval()
        predictions = []
        atoms_counts = []
        
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch)
                for i in range(len(out)):
                    predictions.append(out[i].item())
                    atoms_counts.append(batch.n_atoms[i].item() if hasattr(batch, 'n_atoms') else graphs[i].n_atoms)
        
        # Return predictions with atoms counts for reference
        return list(zip(predictions, atoms_counts))

    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        if self.model is None:
            input_dim = self.graphs[0].x.shape[1] if self.graphs else 7
            self.model = GNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    # Example usage
    dft_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFT_data", "DFT_data")
    print(f"Using DFT data path: {dft_path}")
    
    try:
        # Initialize and prepare data
        ml = MaterialML(dft_path)
        train_loader, val_loader = ml.prepare_data()
        
        # Print data statistics
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
            break  # Only need first batch for statistics
        
        # Train model
        ml.train(train_loader, val_loader)
        
        # Make predictions on validation set
        ml.model.eval()
        print("\nPrediction Examples:")
        with torch.no_grad():
            for batch in val_loader:
                pred = ml.predict(batch)
                print("\nBatch Statistics:")
                print(f"Batch size: {len(pred)} systems")
                
                total_abs_error_per_atom = 0
                total_abs_error = 0
                
                # Print predictions for each graph in the batch
                for i in range(len(pred)):
                    n_atoms = batch.n_atoms[i]
                    
                    # Convert back to total energy for comparison
                    pred_total = pred[i].item() * n_atoms
                    actual_total = batch.y[i].item() * n_atoms
                    
                    # Calculate errors
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
                
                # Print batch summary
                avg_error_per_atom = total_abs_error_per_atom / len(pred)
                avg_error_total = total_abs_error / len(pred)
                print(f"\nBatch Summary:")
                print(f"Average absolute error per atom: {avg_error_per_atom:.4f} eV/atom")
                print(f"Average absolute total error: {avg_error_total:.4f} eV")
                break  # Just show first batch
                
        # Example usage of the new POSCAR-only prediction
        # Uncomment to test prediction from POSCAR files directly
        print("\nTesting prediction from POSCAR files:")
        model_path = os.path.join(ml.models_dir, "best_model.pth")
        if os.path.exists(model_path):
            ml.load_model(model_path)
            
            # Replace with actual paths to your unit cell and supercell POSCAR files
            unit_cell_poscar = "DFT_data/DFT_data/POSCAR/Unit_Super/POSCAR_unitcell.vasp"
            supercell_poscar = "DFT_data/DFT_data/POSCAR/Unit_Super/POSCAR_supercell.vasp"
            
            if os.path.exists(unit_cell_poscar) and os.path.exists(supercell_poscar):
                predictions = ml.predict_from_poscar([unit_cell_poscar, supercell_poscar])
                
                print("\nPOSCAR Prediction Results:")
                print(f"Unit cell energy per atom: {predictions[0][0]:.6f} eV/atom (atoms: {predictions[0][1]})")
                print(f"Supercell energy per atom: {predictions[1][0]:.6f} eV/atom (atoms: {predictions[1][1]})")
                
                # Calculate percent difference
                if abs(predictions[0][0]) > 1e-6:  # Avoid division by zero
                    percent_diff = abs(predictions[1][0] - predictions[0][0]) / abs(predictions[0][0]) * 100
                    print(f"Percent difference: {percent_diff:.4f}%")
            else:
                print("POSCAR files not found. Update the paths to test this feature.")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        
# Normalise energy by  dividing by number of atoms 
# Visualize it 
# VESTA to visualizePOSCAR files , POSCAR have lattice vectorrs line 3-5 are lattice vectors , all below are lattice positions
# POSCAR files for postition OUTCAR output of DFT
# Weights difference for boron cabron
# Energy / Atoms