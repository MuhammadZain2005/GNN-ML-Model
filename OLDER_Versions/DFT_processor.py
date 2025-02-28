import os
import numpy as np
from ase.io import read
import torch
from torch_geometric.data import Data

class DFTProcessor:
    def __init__(self, dft_data_path):
        """Initialize DFT processor with path to data directory"""
        self.dft_data_path = os.path.abspath(dft_data_path)
        
    def read_structure(self, structure_file):
        """Read atomic structure from POSCAR file"""
        try:
            return read(structure_file, format='vasp')
        except Exception as e:
            print(f"Error reading structure file {structure_file}: {e}")
            return None

    def extract_energy_from_outcar(self, outcar_file):
        """Extract final energy from OUTCAR file"""
        try:
            energy = None
            with open(outcar_file, 'r') as f:
                for line in f:
                    if "free  energy   TOTEN" in line:
                        energy = float(line.split()[-2])
            return energy
        except Exception as e:
            print(f"Error extracting energy from OUTCAR {outcar_file}: {e}")
            return None

    def extract_forces_from_outcar(self, outcar_file):
        """Extract forces from OUTCAR file"""
        try:
            forces = []
            with open(outcar_file, 'r') as f:
                lines = f.readlines()
                
            # Find the last TOTAL-FORCE section
            for i in reversed(range(len(lines))):
                if 'TOTAL-FORCE' in lines[i]:
                    # Next line is header, forces start after
                    forces = []
                    for j in range(i+2, len(lines)):
                        if '--------' in lines[j]:
                            break
                        force = list(map(float, lines[j].split()[3:6]))
                        forces.append(force)
                    break
                    
            return np.array(forces) if forces else None
            
        except Exception as e:
            print(f"Error extracting forces from OUTCAR {outcar_file}: {e}")
            return None

    def extract_features(self, structure, forces, energy):
        """Extract atomic features from structure and forces"""
        if structure is None or forces is None or energy is None:
            return None
        
        try:
            atomic_numbers = structure.get_atomic_numbers()
            positions = structure.get_positions()
            
            # Validate data shapes
            if len(atomic_numbers) != len(positions) or len(positions) != len(forces):
                print(f"Mismatch in data dimensions: atoms={len(atomic_numbers)}, positions={len(positions)}, forces={len(forces)}")
                return None
            
            # Basic features: [atomic_number, x, y, z, fx, fy, fz]
            features = np.column_stack([
                atomic_numbers,
                positions,
                forces
            ])
            
            return features, energy
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def build_graph(self, structure, features, energy):
        """Construct graph from atomic structure and features"""
        if features is None:
            return None

        try:
            positions = structure.get_positions()
            
            # Calculate pairwise distances
            n_atoms = len(positions)
            distances = np.zeros((n_atoms, n_atoms))
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j:
                        diff = positions[i] - positions[j]
                        distances[i,j] = np.sqrt(np.sum(diff**2))
            
            # Create edges based on distance cutoff
            cutoff = 3.0  # Angstrom
            edge_index = []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j and distances[i,j] < cutoff:
                        edge_index.append([i, j])
            
            if not edge_index:
                print("No edges found within cutoff distance")
                return None
                
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Create node features and target energy
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor([energy], dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, y=y)
            
        except Exception as e:
            print(f"Error building graph: {e}")
            return None
    
    def process_directory(self):
        """Process all DFT calculations in directory recursively"""
        graphs = []
        print(f"Processing DFT data from: {self.dft_data_path}")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.dft_data_path):
            # Check for required files
            poscar_path = os.path.join(root, 'POSCAR')
            outcar_path = os.path.join(root, 'OUTCAR')
            
            if not (os.path.exists(poscar_path) and os.path.exists(outcar_path)):
                continue
                
            print(f"\nProcessing directory: {root}")
            
            try:
                # Read structure from POSCAR
                structure = self.read_structure(poscar_path)
                if structure is None:
                    continue
                    
                # Extract forces and energy from OUTCAR
                forces = self.extract_forces_from_outcar(outcar_path)
                energy = self.extract_energy_from_outcar(outcar_path)
                
                if forces is None or energy is None:
                    print(f"Could not extract forces or energy from {outcar_path}")
                    continue
                
                # Extract features
                result = self.extract_features(structure, forces, energy)
                if result is None:
                    continue
                    
                features, energy = result
                
                # Build graph
                graph = self.build_graph(structure, features, energy)
                if graph is not None:
                    graphs.append(graph)
                    print(f"Successfully processed {root}")
                    print(f"Number of atoms: {len(structure)}")
                    print(f"Energy: {energy:.4f} eV")
                    
            except Exception as e:
                print(f"Error processing directory {root}: {e}")
                continue
                    
        print(f"\nTotal graphs created: {len(graphs)}")
        return graphs
