import torch
from torch_geometric.loader import DataLoader
from DFT_processor_2_Zain import DFTProcessor
from gnn_model import EnhancedGNNModel, GNNModel, EquivariantGNNModel
import os
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class MaterialML:
    def __init__(self, dft_data_path, use_enhanced_model=True, use_equivariant_model=False, focus_hard=False):
        """
        Initialize with path to DFT data
        
        Args:
            dft_data_path: Path to DFT calculation data
            use_enhanced_model: Whether to use the enhanced model with edge attributes
            use_equivariant_model: Whether to use the equivariant model with force prediction
            focus_hard: Whether to focus on hard examples during training
        """
        self.dft_data_path = dft_data_path
        self.use_enhanced_model = use_enhanced_model
        self.use_equivariant_model = use_equivariant_model
        self.focus_hard = focus_hard  # New flag for focusing on hard examples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Increased epochs, LR scheduling, Early stopping, Periodic checkpointing
    def train(self, train_loader, val_loader, hidden_dim=128, lr=0.0001, epochs=500, w_forces=0.0):
        """
        Train GNN model with optional force prediction and hard example emphasis
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            hidden_dim: Size of hidden layers
            lr: Learning rate
            epochs: Number of training epochs
            w_forces: Weight for force loss (0.0 to disable force prediction)
        """
        input_dim = train_loader.dataset[0].x.shape[1]
        output_dim = 1

        # Model initialization
        if self.use_equivariant_model:
            self.model = EquivariantGNNModel(input_dim, hidden_dim, output_dim)
            model_path = os.path.join(self.models_dir, "best_equivariant_model.pth")
            print("Using EquivariantGNNModel with dual energy/force prediction")
        elif self.use_enhanced_model:
            self.model = EnhancedGNNModel(input_dim, hidden_dim, output_dim)
            model_path = os.path.join(self.models_dir, "best_enhanced_model.pth")
            print("Using EnhancedGNNModel with edge attributes for PBC")
        else:
            self.model = GNNModel(input_dim, hidden_dim, output_dim)
            model_path = os.path.join(self.models_dir, "best_model.pth")
            print("Using original GNNModel")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-5)
        energy_criterion = torch.nn.MSELoss()
        force_criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50

        train_losses = []
        val_losses = []
        force_losses = []

        use_forces = w_forces > 0 and self.use_equivariant_model

        print(f"Training with force prediction: {use_forces}")
        if use_forces:
            print(f"Force loss weight: {w_forces}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_force_loss = 0
            batch_count = 0

            for batch in train_loader:
                optimizer.zero_grad()
                
                if self.use_equivariant_model:
                    # Dual-head prediction (energy and forces)
                    energy_pred, force_pred, pos = self.model(batch)
                    energy_pred = energy_pred.squeeze(-1)
                    
                    # Energy loss
                    energy_loss = energy_criterion(energy_pred, batch.y)
                    
                    # Initialize total loss with energy loss
                    loss = energy_loss
                    
                    # Add force loss if enabled
                    if use_forces:
                        # Extract forces from input features
                        actual_forces = batch.x[:, 4:7]
                        
                        # Compute force loss
                        force_loss = force_criterion(force_pred, actual_forces)
                        total_force_loss += force_loss.item()
                        
                        # Add weighted force loss to total loss
                        loss = energy_loss + w_forces * force_loss
                else:
                    # Legacy models (energy prediction only)
                    out = self.model(batch).squeeze(-1)
                
                    if self.focus_hard:
                        # Calculate absolute error
                        error = torch.abs(out - batch.y)
                        threshold = error.mean().item()
                        mask = (error > threshold).float()

                        # Use masked MSE
                        if mask.sum().item() > 0:
                            loss = ((out - batch.y) ** 2 * mask).sum() / (mask.sum() + 1e-6)
                        else:
                            loss = energy_criterion(out, batch.y)
                    else:
                        loss = energy_criterion(out, batch.y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            avg_train_loss = total_loss / batch_count
            train_losses.append(avg_train_loss)
            
            # Log force loss if using force prediction
            if use_forces:
                avg_force_loss = total_force_loss / batch_count
                force_losses.append(avg_force_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            val_force_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if self.use_equivariant_model:
                        energy_pred, force_pred, _ = self.model(batch)
                        energy_pred = energy_pred.squeeze(-1)
                        val_loss += energy_criterion(energy_pred, batch.y).item()
                        
                        if use_forces:
                            actual_forces = batch.x[:, 4:7]
                            val_force_loss += force_criterion(force_pred, actual_forces).item()
                    else:
                        out = self.model(batch).squeeze(-1)
                        val_loss += energy_criterion(out, batch.y).item()
                    val_batch_count += 1
                    
            avg_val_loss = val_loss / val_batch_count
            val_losses.append(avg_val_loss)
            
            # Log training progress
            log_str = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            if use_forces:
                avg_val_force_loss = val_force_loss / val_batch_count
                log_str += f" | Force Loss: {avg_force_loss:.6f} | Val Force Loss: {avg_val_force_loss:.6f}"
            print(log_str)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(model_path)
                print(f"New best model saved with val loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if epoch % 50 == 0:
                checkpoint_path = os.path.join(self.models_dir, f"model_checkpoint_epoch_{epoch}.pth")
                self.save_model(checkpoint_path)

        self.load_model(model_path)
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        if use_forces and force_losses:
            # Normalize force losses to be on similar scale as energy losses
            max_energy = max(max(train_losses), max(val_losses))
            max_force = max(force_losses) if force_losses else 1.0
            scale_factor = max_energy / max_force if max_force > 0 else 1.0
            scaled_force_losses = [fl * scale_factor for fl in force_losses]
            plt.plot(scaled_force_losses, label='Force Loss (Scaled)')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plot_path = os.path.join(self.models_dir, 'training_plot.png')
        plt.savefig(plot_path)
        print(f"Training plot saved to {plot_path}")

    def predict(self, data=None):
        """
        Make a prediction for a graph. Returns different outputs based on model type.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        
        if self.use_equivariant_model:
            # Return full prediction dict with energy and forces
            with torch.set_grad_enabled(True):
                return self.model.predict(data)
        else:
            # Legacy model returns energy only
            with torch.no_grad():
                return self.model(data)
    
    def predict_forces(self, data):
        """
        Predict forces for a graph using either direct prediction or autograd.
        
        Args:
            data: PyG Data object
            
        Returns:
            Dictionary with direct and autograd forces
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        if not self.use_equivariant_model:
            raise ValueError("Force prediction requires equivariant model")
            
        # Use the model's predict method which handles both energy and force prediction
        return self.model.predict(data)
    
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
        
        results = []
        for batch in loader:
            predictions = self.predict(batch)
            
            if self.use_equivariant_model:
                # For equivariant model, extract energy and forces
                energies = predictions['energy']
                forces = predictions['forces_autograd']  # Use autograd forces
                
                for i in range(len(energies)):
                    energy_per_atom = energies[i].item()
                    n_atoms = batch.n_atoms[i].item() if hasattr(batch, 'n_atoms') else graphs[i].n_atoms
                    total_energy = energy_per_atom * n_atoms
                    
                    # Get forces for this graph
                    if len(graphs) > 1:
                        # Get correct slice of forces for this graph
                        start_idx = sum(g.x.shape[0] for g in graphs[:i]) if i > 0 else 0
                        end_idx = start_idx + graphs[i].x.shape[0]
                        graph_forces = forces[start_idx:end_idx]
                    else:
                        graph_forces = forces
                        
                    results.append({
                        'energy_per_atom': energy_per_atom,
                        'total_energy': total_energy,
                        'n_atoms': n_atoms,
                        'forces': graph_forces.detach().cpu().numpy()
                    })
            else:
                # Legacy behavior for non-equivariant models
                for i in range(len(predictions)):
                    energy_per_atom = predictions[i].item()
                    n_atoms = batch.n_atoms[i].item() if hasattr(batch, 'n_atoms') else graphs[i].n_atoms
                    total_energy = energy_per_atom * n_atoms
                    results.append({
                        'energy_per_atom': energy_per_atom,
                        'total_energy': total_energy,
                        'n_atoms': n_atoms
                    })
                
        return results

    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
            
        # Determine the type of model to load
        if "equivariant" in path:
            use_equivariant = True
            use_enhanced = False
        elif "enhanced" in path:
            use_equivariant = False
            use_enhanced = True
        else:
            use_equivariant = self.use_equivariant_model
            use_enhanced = self.use_enhanced_model
            
        if self.model is None:
            input_dim = self.graphs[0].x.shape[1] if self.graphs else 7
            if use_equivariant:
                self.model = EquivariantGNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1)
            elif use_enhanced:
                self.model = EnhancedGNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1)
            else:
                self.model = GNNModel(input_dim=input_dim, hidden_dim=128, output_dim=1)
                
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    dft_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFT_data", "DFT_data")
    print(f"Using DFT data path: {dft_path}")
    
    try:
        # Create model with equivariant architecture for force prediction
        ml = MaterialML(dft_path, use_enhanced_model=False, use_equivariant_model=True, focus_hard=True)
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
        
        # Train with force prediction (w_forces=100 means force loss is weighted 100x energy loss)
        ml.train(train_loader, val_loader, w_forces=100.0)
        
        # Predictions on validation set
        ml.model.eval()
        print("\nPrediction Examples:")
        with torch.set_grad_enabled(True):  # Need gradients for force prediction
            for batch in val_loader:
                # Get predictions (now includes forces)
                predictions = ml.predict(batch)
                
                energy_pred = predictions['energy']
                forces_direct = predictions['forces_direct']
                forces_autograd = predictions['forces_autograd']
                
                print("\nBatch Statistics:")
                print(f"Batch size: {len(energy_pred)} systems")
                
                # Process energy predictions
                total_abs_error_per_atom = 0
                total_abs_error = 0
                
                for i in range(len(energy_pred)):
                    n_atoms = batch.n_atoms[i]
                    pred_total = energy_pred[i].item() * n_atoms
                    actual_total = batch.y[i].item() * n_atoms
                    error_per_atom = abs(energy_pred[i].item() - batch.y[i].item())
                    error_total = abs(pred_total - actual_total)
                    total_abs_error_per_atom += error_per_atom
                    total_abs_error += error_total
                    
                    print(f"\nGraph {i+1} in batch:")
                    print(f"Number of atoms: {n_atoms}")
                    print(f"Predicted energy per atom: {energy_pred[i].item():.4f} eV/atom")
                    print(f"Actual energy per atom: {batch.y[i].item():.4f} eV/atom")
                    print(f"Predicted total energy: {pred_total:.4f} eV")
                    print(f"Actual total energy: {actual_total:.4f} eV")
                    print(f"Absolute error per atom: {error_per_atom:.4f} eV/atom")
                    print(f"Absolute total error: {error_total:.4f} eV")
                    
                    # Extract actual forces for this graph
                    if i == 0:  # Just report first graph's force stats to keep output manageable
                        # Extract indices for this graph's atoms
                        start_idx = 0
                        end_idx = n_atoms.item()
                        
                        # Extract actual forces from input features
                        actual_forces = batch.x[start_idx:end_idx, 4:7]
                        
                        # Extract predicted forces
                        pred_forces_direct = forces_direct[start_idx:end_idx]
                        pred_forces_autograd = forces_autograd[start_idx:end_idx]
                        
                        # Calculate force errors
                        direct_force_error = torch.mean(torch.abs(pred_forces_direct - actual_forces)).item()
                        autograd_force_error = torch.mean(torch.abs(pred_forces_autograd - actual_forces)).item()
                        
                        print("\nForce Prediction (sample of 3 atoms):")
                        for j in range(min(3, n_atoms.item())):
                            print(f"  Atom {j+1}:")
                            print(f"    Actual:   [{actual_forces[j, 0]:.4f}, {actual_forces[j, 1]:.4f}, {actual_forces[j, 2]:.4f}]")
                            print(f"    Direct:   [{pred_forces_direct[j, 0]:.4f}, {pred_forces_direct[j, 1]:.4f}, {pred_forces_direct[j, 2]:.4f}]")
                            print(f"    Autograd: [{pred_forces_autograd[j, 0]:.4f}, {pred_forces_autograd[j, 1]:.4f}, {pred_forces_autograd[j, 2]:.4f}]")
                        
                        print(f"\nAverage force prediction errors:")
                        print(f"  Direct method:   {direct_force_error:.4f} eV/Å")
                        print(f"  Autograd method: {autograd_force_error:.4f} eV/Å")
                
                avg_error_per_atom = total_abs_error_per_atom / len(energy_pred)
                avg_error_total = total_abs_error / len(energy_pred)
                print(f"\nBatch Summary:")
                print(f"Average absolute error per atom: {avg_error_per_atom:.4f} eV/atom")
                print(f"Average absolute total error: {avg_error_total:.4f} eV")
                break
        
        # === Energy Parity Plot ===
        dft_energies = batch.y.cpu().numpy()
        pred_energies = energy_pred.cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.scatter(dft_energies, pred_energies, alpha=0.6, edgecolors='k', s=60)
        plt.plot([min(dft_energies), max(dft_energies)],
                [min(dft_energies), max(dft_energies)], 'r--', label='x = y')
        plt.xlabel("DFT Energy per Atom (eV)")
        plt.ylabel("Predicted (GNN) Energy per Atom (eV)")
        plt.title("Energy Parity Plot")
        plt.legend()
        plt.grid(True)

        n_points = len(dft_energies)
        plt.text(
            0.95, 0.05,
            f"N = {n_points}",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))
        # Compute R² and MSE
        r2 = r2_score(dft_energies, pred_energies)
        mse = mean_squared_error(dft_energies, pred_energies)

        # Add R² and MSE to top-left corner
        plt.text(
            0.05, 0.95,
            f"$R^2$ = {r2:.4f}\nMSE = {mse:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha='left',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))

        parity_plot_path = os.path.join(ml.models_dir, 'energy_parity_plot.png')
        plt.savefig(parity_plot_path)
        print(f"Energy parity plot saved to {parity_plot_path}")
        
        # Force comparison plot (if using equivariant model)
        if ml.use_equivariant_model:
            # Extract data for the force plot
            # Just using the first graph in the batch for simplicity
            start_idx = 0
            end_idx = batch.n_atoms[0].item()
            
            # Flatten forces to get components
            actual_forces_flat = batch.x[start_idx:end_idx, 4:7].view(-1).cpu().numpy()
            pred_forces_flat = forces_autograd[start_idx:end_idx].view(-1).detach().cpu().numpy()
            
            plt.figure(figsize=(8, 8))
            plt.scatter(actual_forces_flat, pred_forces_flat, alpha=0.6, edgecolors='k', s=40)
            min_val = min(actual_forces_flat.min(), pred_forces_flat.min())
            max_val = max(actual_forces_flat.max(), pred_forces_flat.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
            plt.xlabel("DFT Force Components (eV/Å)")
            plt.ylabel("Predicted Force Components (eV/Å)")
            plt.title("Force Parity Plot")
            plt.legend()
            plt.grid(True)
            
            # Compute R² and MSE for forces
            force_r2 = r2_score(actual_forces_flat, pred_forces_flat)
            force_mse = mean_squared_error(actual_forces_flat, pred_forces_flat)
            force_mae = np.mean(np.abs(actual_forces_flat - pred_forces_flat))
            
            # Add stats to plot
            plt.text(
                0.05, 0.95,
                f"$R^2$ = {force_r2:.4f}\nMAE = {force_mae:.4f} eV/Å",
                transform=plt.gca().transAxes,
                fontsize=12,
                ha='left',
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))
                
            force_plot_path = os.path.join(ml.models_dir, 'force_parity_plot.png')
            plt.savefig(force_plot_path)
            print(f"Force parity plot saved to {force_plot_path}")

        # Test POSCAR-only predictions
        print("\nTesting prediction from POSCAR files:")
        model_path = os.path.join(ml.models_dir, "best_equivariant_model.pth")
        if os.path.exists(model_path):
            ml.load_model(model_path)
            unit_cell_poscar = "DFT_data/DFT_data/POSCAR/Unit_Super/POSCAR_unitcell.vasp"
            supercell_poscar = "DFT_data/DFT_data/POSCAR/Unit_Super/POSCAR_supercell.vasp"
            if os.path.exists(unit_cell_poscar) and os.path.exists(supercell_poscar):
                predictions = ml.predict_from_poscar([unit_cell_poscar, supercell_poscar])
                print("\nPOSCAR Prediction Results:")
                
                # Extract unit cell predictions
                unit_pred = predictions[0]
                unit_energy_per_atom = unit_pred['energy_per_atom']
                unit_total_energy = unit_pred['total_energy']
                unit_n_atoms = unit_pred['n_atoms']
                
                # Extract supercell predictions
                super_pred = predictions[1]
                super_energy_per_atom = super_pred['energy_per_atom']
                super_total_energy = super_pred['total_energy']
                super_n_atoms = super_pred['n_atoms']
                
                print(f"Unit cell energy per atom: {unit_energy_per_atom:.6f} eV/atom (atoms: {unit_n_atoms})")
                print(f"Supercell energy per atom: {super_energy_per_atom:.6f} eV/atom (atoms: {super_n_atoms})")
                
                if 'forces' in unit_pred:
                    print("\nUnit cell forces (sample):")
                    forces = unit_pred['forces']
                    for i in range(min(3, len(forces))):
                        print(f"  Atom {i+1}: [{forces[i, 0]:.4f}, {forces[i, 1]:.4f}, {forces[i, 2]:.4f}] eV/Å")
                
                if abs(unit_energy_per_atom) > 1e-6:
                    percent_diff = abs(super_energy_per_atom - unit_energy_per_atom) / abs(unit_energy_per_atom) * 100
                    print(f"Percent difference: {percent_diff:.4f}%")
            else:
                print("POSCAR files not found. Update the paths to test this feature.")
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())


# Energy Parity Plot x= dft energies y= GNN energies (predicted) x=y line - DONE

# Try cutoff functions sinh tanh, 6.5 highest distance