import torch
from torch_geometric.loader import DataLoader
from DFT_processor import DFTProcessor
from gnn_model import GNNModel
import os

class MaterialML:
    def __init__(self, dft_data_path):
        """Initialize with path to DFT data"""
        self.processor = DFTProcessor(dft_data_path)
        self.model = None
        
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
    def prepare_data(self):
        """Process DFT data and prepare training set"""
        graphs = self.processor.process_directory()
        
        if not graphs:
            raise ValueError(f"""
No graphs were created. Possible reasons:
1. Incorrect DFT data path: {self.processor.dft_data_path}
2. Missing required files (POSCAR and OUTCAR)
3. Errors in processing the DFT files
Check the debug output above for more details.
""")
        
        # Energy values are already included in the graphs from DFT_processor
        print(f"Number of graphs loaded: {len(graphs)}")
        print("\nData Statistics:")
        energies = torch.stack([g.y for g in graphs])
        print(f"Energy range: {energies.min():.4f} to {energies.max():.4f} eV")
        print(f"Mean energy: {energies.mean():.4f} eV")
        print(f"Energy std: {energies.std():.4f} eV")
        
        # Split into training and validation sets (80-20 split)
        n_train = int(0.8 * len(graphs))
        train_graphs = graphs[:n_train]
        val_graphs = graphs[n_train:]
        
        print(f"\nSplit sizes:")
        print(f"Training graphs: {len(train_graphs)}")
        print(f"Validation graphs: {len(val_graphs)}")
        
        # Create data loaders with smaller batch size
        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
        
        return train_loader, val_loader

    def train(self, train_loader, val_loader, hidden_dim=64, lr=0.001, epochs=100):
        """Train GNN model"""
        # Get dimensions from data
        input_dim = train_loader.dataset[0].x.shape[1]  # Number of node features
        output_dim = 1  # Energy prediction (scalar)
        
        # Initialize model
        self.model = GNNModel(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        print("\nTraining Model:")
        print(f"Input dimensions: {input_dim}")
        print(f"Hidden dimensions: {hidden_dim}")
        print(f"Output dimensions: {output_dim}")
        print(f"Learning rate: {lr}")
        print(f"Number of epochs: {epochs}\n")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Path for best model
        best_model_path = os.path.join(self.models_dir, "best_model.pth")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = self.model(batch)
                    val_loss += criterion(out, batch.y).item()
            avg_val_loss = val_loss / len(val_loader)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Training Loss: {avg_train_loss:.6f}")
                print(f"Validation Loss: {avg_val_loss:.6f}\n")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model(best_model_path)
                print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_model(best_model_path)
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        print(f"Best model saved at: {best_model_path}")
        
    def predict(self, data):
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        with torch.no_grad():
            return self.model(data)
    
    def save_model(self, path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load pre-trained model"""
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        if self.model is None:
            raise ValueError("Model not initialized")
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
        
        # Train model
        ml.train(train_loader, val_loader)
        
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
                
    except Exception as e:
        print(f"Error: {str(e)}")
