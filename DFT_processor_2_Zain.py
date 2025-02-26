import os
import numpy as np
import torch
from torch_geometric.data import Data
from openpyxl import Workbook

class DFTProcessor:
    def __init__(self, dft_data_path):
        """Initialize DFT processor with path to data directory"""
        self.dft_data_path = os.path.abspath(dft_data_path)
        self.graphs = []  # Store processed graphs
        
    def __len__(self):
        """Make the processor lennable"""
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """Make the processor indexable"""
        return self.graphs[idx]
        
    def read_POSCAR(self, file_path):
        """Read lattice vectors, atom positions, and atom counts from a POSCAR file"""
        lattice_vectors = []
        atom_positions = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[2:5]:
                lattice_vectors.append([float(val) for val in line.split()])
            atom_counts = lines[6].split()
            n_boron = int(atom_counts[0])
            n_carbon = int(atom_counts[1])
            total_atoms = n_boron + n_carbon
            for line in lines[8:8 + total_atoms]:
                atom_positions.append([float(val) for val in line.split()])
        return lattice_vectors, atom_positions, n_boron, n_carbon, total_atoms

    def extract_OUTCAR(self, file_path):
        """Extract the final energy and forces from an OUTCAR file"""
        final_energy = None
        final_forces = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):
                if "free  energy   TOTEN" in line:
                    final_energy = float(line.split()[-2])
                    break
            start_index = None
            for i, line in enumerate(reversed(lines)):
                if "POSITION                                       TOTAL-FORCE (eV/Angst)" in line:
                    start_index = len(lines) - i + 1
                    break
            if start_index is not None:
                for line in lines[start_index:]:
                    if not line.strip():
                        break
                    if not all(char == '-' for char in line.strip()):
                        values = line.split()[-3:]
                        forces = [float(val) for val in values]
                        final_forces.append(forces)
        return final_energy, final_forces

    def build_graph(self, atom_positions, atom_types, forces, energy, cutoff=3.0):
        """Construct graph from atomic structure and features with normalized energy"""
        try:
            positions = np.array(atom_positions)
            n_atoms = len(positions)

            # Calculate pairwise distances and create edges
            edge_index = []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j:
                        diff = positions[i] - positions[j]
                        distance = np.sqrt(np.sum(diff**2))
                        if distance < cutoff:
                            edge_index.append([i, j])

            if not edge_index:
                print("No edges found within cutoff distance")
                return None

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Create node features: [atomic_number, x, y, z, fx, fy, fz]
            features = []
            for i in range(n_atoms):
                atomic_number = 5 if atom_types[i] == "B" else 6  # B=5, C=6
                features.append([
                    atomic_number,
                    *positions[i],
                    *forces[i]
                ])

            x = torch.tensor(features, dtype=torch.float)

            # Normalize energy by number of atoms
            normalized_energy = energy / n_atoms
            y = torch.tensor([normalized_energy], dtype=torch.float)

            # Store both normalized and raw energy in the graph
            # This allows us to convert back if needed
            graph = Data(x=x, edge_index=edge_index, y=y)
            graph.raw_energy = torch.tensor([energy], dtype=torch.float)
            graph.n_atoms = n_atoms

            return graph

        except Exception as e:
            print(f"Error building graph: {e}")
            return None
    
    def process_directory(self):
        """Process all DFT calculations in directory recursively"""
        self.graphs = []  # Reset graphs list
        
        # Create Excel workbook for original functionality
        wb = Workbook()
        ws = wb.active
        ws.title = "Data"
        
        # Conversion factors
        angstrom_to_bohr = 1.88973
        ev_to_ha = 1 / 27.211386
        
        for root, dirs, files in os.walk(self.dft_data_path):
            poscar_file = os.path.join(root, 'POSCAR')
            outcar_file = os.path.join(root, 'OUTCAR')
            
            if not (os.path.exists(poscar_file) and os.path.exists(outcar_file)):
                continue
                
            print(f"\nProcessing directory: {root}")
            
            try:
                # Read structure data
                lattice_vectors, atom_positions, n_boron, n_carbon, total_atoms = self.read_POSCAR(poscar_file)
                final_energy, final_forces = self.extract_OUTCAR(outcar_file)
                
                if final_energy is None or not final_forces:
                    print(f"Could not extract data from {outcar_file}")
                    continue
                
                # Create atom types list
                atom_types = ["B"] * n_boron + ["C"] * n_carbon
                
                # Build graph for ML
                graph = self.build_graph(atom_positions, atom_types, final_forces, final_energy)
                if graph is not None:
                    self.graphs.append(graph)
                
                # Original Excel output functionality
                ws.append(["begin", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                
                for vector in lattice_vectors:
                    bohr_vector = [val * angstrom_to_bohr for val in vector]
                    ws.append(["lattice", *bohr_vector, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                
                final_forces_ha_bohr = [[val * ev_to_ha/angstrom_to_bohr for val in forces] for forces in final_forces]
                
                for i, position in enumerate(atom_positions, start=1):
                    bohr_position = [val * angstrom_to_bohr for val in position]
                    atom_type = atom_types[i-1]
                    ws.append(["atom", *bohr_position, atom_type, 0, 0, *final_forces_ha_bohr[i-1], "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                
                final_energy_ha = final_energy * ev_to_ha
                ws.append(["energy", final_energy_ha, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                ws.append(["charge", "0.00", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                ws.append(["end", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                
            except Exception as e:
                print(f"Error processing directory {root}: {e}")
                continue
        
        # Save Excel file
        output_file = os.path.join(os.path.dirname(self.dft_data_path), 'data.csv')
        wb.save(output_file)
        print(f"\nData saved to {output_file}")
        print(f"Total graphs created: {len(self.graphs)}")
        
        return self.graphs