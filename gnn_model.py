import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, BatchNorm, MessagePassing
from torch.nn import Sequential, Linear, ReLU, Parameter
import e3nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

# Custom GNN layer that incorporates edge features/distances
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='add')
        
        # Node feature transformation
        self.node_mlp = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        
        # Edge feature transformation
        self.edge_mlp = Sequential(
            Linear(out_channels * 2 + 1, out_channels), # +1 for edge distance
            ReLU(),
            Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        
        # Transform node features
        x = self.node_mlp(x)
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Create edge messages based on the source node, target node, and edge features
        edge_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(edge_features)

    def update(self, aggr_out, x):
        # Update node embeddings
        return aggr_out + x 

# Simplified equivariant layer to avoid complex irreps operations
class SimpleEquivariantGNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        """
        Simplified equivariant model for Week 1 implementation testing.
        Uses position gradients but with simplified architecture.
        
        Args:
            input_dim: Number of node features (scalar features)
            hidden_dim: Size of hidden layers
            output_dim: Output size (1 for energy prediction)
        """
        super(SimpleEquivariantGNNModel, self).__init__()
        
        # Regular GNN layers for scalar features
        self.conv1 = EdgeConv(input_dim, hidden_dim)
        self.conv2 = EdgeConv(hidden_dim, hidden_dim)
        
        # Batch Normalization
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        # Shared linear layer for both heads
        self.shared_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Energy prediction head (scalar prediction)
        self.energy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
        # Force prediction head (vector prediction per atom)
        self.force_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3)  # 3D force vector per atom
        )
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(p=0.2)
        
    def forward(self, data):
        """
        Forward pass through the model.
        Returns both energy prediction and force prediction.
        """
        # Ensure position gradient tracking for autograd forces
        pos = data.x[:, 1:4].clone()
        pos.requires_grad_(True)
        
        # Replace position in features with gradable version
        x = torch.cat([data.x[:, 0:1], pos, data.x[:, 4:]], dim=1)
        
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device) if batch is None else batch
        
        # Apply convolutional layers
        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)))
        x2 = self.dropout(x2) + x1  # Residual connection
        
        # Apply shared layer
        shared_features = F.relu(self.shared_layer(x2))
        
        # Per-node features for force prediction
        node_features = shared_features
        
        # Global pooling for energy prediction
        global_features = global_mean_pool(shared_features, batch)
        
        # Energy prediction (scalar)
        energy_pred = self.energy_head(global_features)
        
        # Force prediction (vector per atom)
        force_pred = self.force_head(node_features)
        
        # Return both predictions
        return energy_pred, force_pred, pos
    
    def compute_autograd_forces(self, energy, pos):
        """
        Compute forces as the negative gradient of energy with respect to positions.
        
        Args:
            energy: Predicted energy
            pos: Atomic positions with gradients tracked
            
        Returns:
            forces: Predicted forces from autograd
        """
        forces = -torch.autograd.grad(
            energy.sum(), 
            pos, 
            create_graph=True, 
            retain_graph=True
        )[0]
        return forces
    
    def predict(self, data):
        """
        Prediction mode with both direct force prediction and autograd forces.
        """
        self.eval()
        with torch.set_grad_enabled(True):  # Need gradients for autograd forces
            energy_pred, force_pred, pos = self(data)
            autograd_forces = self.compute_autograd_forces(energy_pred, pos)
            
            return {
                'energy': energy_pred,
                'forces_direct': force_pred,
                'forces_autograd': autograd_forces,
                'positions': pos
            }

# For now, set EquivariantGNNModel as an alias to SimpleEquivariantGNNModel
# This will be replaced with proper equivariant implementation in later weeks
EquivariantGNNModel = SimpleEquivariantGNNModel

class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Enhanced GNN model that uses edge attributes (distances) and proper PBCs.
        
        Args:
            input_dim: Number of node features
            hidden_dim: Size of hidden layers
            output_dim: Output size (1 for energy prediction)
        """
        super(EnhancedGNNModel, self).__init__()

        # GNN Layers with EdgeConv to use distance information
        self.conv1 = EdgeConv(input_dim, hidden_dim)
        self.conv2 = EdgeConv(hidden_dim, hidden_dim)
        self.conv3 = EdgeConv(hidden_dim, hidden_dim)

        # Batch Normalization
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # Fully Connected Layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout for Regularization
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        """
        Forward pass through the model.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device) if batch is None else batch

        # First GNN layer with edge attributes
        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x1 = self.dropout(x1)

        # Second GNN layer with edge attributes and residual connection
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)))
        x2 = self.dropout(x2) + x1 

        # Third GNN layer with edge attributes and residual connection
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index, edge_attr)))
        x3 = self.dropout(x3) + x2  

        # Global Mean Pooling
        x = global_mean_pool(x3, batch)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, data):
        """
        Prediction mode.
        """
        self.eval()
        with torch.no_grad():
            return self(data)


# Keep the original GNN model for backward compatibility
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Original GNN model with residual connections and better pooling.
        Args:
            input_dim: Number of node features
            hidden_dim: Size of hidden layers
            output_dim: Output size (1 for energy prediction)
        """
        super(GNNModel, self).__init__()

        # GNN Layers (Using GIN for better feature extraction)
        self.conv1 = GINConv(torch.nn.Linear(input_dim, hidden_dim))
        self.conv2 = GINConv(torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv3 = GINConv(torch.nn.Linear(hidden_dim, hidden_dim))

        # Batch Normalization
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # Fully Connected Layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout for Regularization
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        """
        Forward pass through the model.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device) if batch is None else batch

        # First GNN layer with residual connection
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)

        # Second GNN layer with residual connection
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2) + x1  # Residual connection

        # Third GNN layer with residual connection
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = self.dropout(x3) + x2  # Residual connection

        # Global Mean Pooling
        x = global_mean_pool(x3, batch)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, data):
        """
        Prediction mode.
        """
        self.eval()
        with torch.no_grad():
            return self(data)