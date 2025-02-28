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
        
    @staticmethod
    def read_POSCAR(file_path):
        """
        Reads a POSCAR file and returns the lattice vectors, atomic positions,
        and atom counts. This version applies the scaling factor properly.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Line 2 contains the scaling factor
        scale = float(lines[1].strip())
        
        # Lines 3-5: Read lattice vectors and apply scaling
        lattice_vectors = np.array([[float(x) for x in line.split()] for line in lines[2:5]]) * scale
        
        # Line 7 contains atom counts (assuming format: "n_B n_C")
        atom_counts = lines[6].split()
        n_boron = int(atom_counts[0])
        n_carbon = int(atom_counts[1])
        total_atoms = n_boron + n_carbon
        
        # Read atomic positions (assume they start at line 9)
        pos_start = 8
        positions = []
        for line in lines[pos_start:pos_start + total_atoms]:
            positions.append([float(x) for x in line.split()[:3]])
        atom_positions = np.array(positions)
        
        # Check if the positions are given in "Direct" (fractional) coordinates
        coord_type = lines[pos_start - 1].strip().lower()
        if "direct" in coord_type:
            # Convert fractional positions to Cartesian using lattice vectors
            atom_positions = np.dot(atom_positions, lattice_vectors)
        
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

    def build_graph(self, atom_positions, atom_types, forces, energy, lattice_vectors, cutoff=3.15):
        """
        Constructs a torch_geometric Data object from the atomic structure,
        using enhanced periodic boundary conditions to compute edges.
        Now includes connections to atoms in neighboring periodic cells.
        """
        try:
            positions = np.array(atom_positions)
            n_atoms = len(positions)
            edge_list = []
            edge_attrs = []
            
            # Generate offsets for nearest neighbor cells (including the original cell)
            offsets = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        offsets.append(np.array([i, j, k]))
            
            # Check distances including periodic images
            for i in range(n_atoms):
                for j in range(n_atoms):  # Consider all atom pairs, including self under PBC
                    if i == j and np.array_equal(offsets[0], [0, 0, 0]):
                        continue  # Skip self-interaction in the original cell
                        
                    # Check atom j and its periodic images
                    for offset in offsets:
                        if i == j and np.array_equal(offset, [0, 0, 0]):
                            continue  # Skip self in original cell
                            
                        # Calculate position of atom j in this periodic image
                        image_pos = positions[j] + np.dot(offset, lattice_vectors)
                        
                        # Calculate direct distance
                        diff = positions[i] - image_pos
                        distance = np.linalg.norm(diff)
                        
                        if distance < cutoff:
                            edge_list.append([i, j])
                            # Store distance as edge attribute
                            edge_attrs.append([distance])
            
            if not edge_list:
                print("No edges found within cutoff distance")
                return None
                
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
            # Build node features: [atomic_number, x, y, z, fx, fy, fz]
            features = []
            for i in range(n_atoms):
                atomic_number = 5 if atom_types[i] == "B" else 6  # B=5, C=6
                features.append([atomic_number, *positions[i], *forces[i]])
            x = torch.tensor(features, dtype=torch.float)
            
            # Normalize energy by number of atoms
            normalized_energy = energy / n_atoms
            y = torch.tensor([normalized_energy], dtype=torch.float)
            
            # Create the Data object with edge attributes
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graph.raw_energy = torch.tensor([energy], dtype=torch.float)
            graph.n_atoms = n_atoms
            
            # Store lattice vectors for possible later use
            graph.lattice = torch.tensor(lattice_vectors, dtype=torch.float)
            
            return graph
        except Exception as e:
            print(f"Error building graph: {e}")
            return None

    @staticmethod
    def minimum_image_distance(pos_i, pos_j, lattice_vectors):
        """
        Calculate the minimum image distance between two atoms using periodic boundary conditions.
        
        Parameters:
            pos_i (np.array): Cartesian coordinates of atom i.
            pos_j (np.array): Cartesian coordinates of atom j.
            lattice_vectors (np.array): 3x3 matrix of lattice vectors.
            
        Returns:
            distance (float): The minimum image distance.
            cart_diff (np.array): The minimum image difference vector.
        """
        # Compute the raw difference
        diff = pos_i - pos_j
        # Convert to fractional coordinates using the inverse of the lattice matrix
        inv_lattice = np.linalg.inv(lattice_vectors)
        fractional_diff = np.dot(inv_lattice, diff)
        # Wrap fractional differences into [-0.5, 0.5)
        fractional_diff = fractional_diff - np.round(fractional_diff)
        # Convert back to Cartesian coordinates
        cart_diff = np.dot(lattice_vectors, fractional_diff)
        distance = np.linalg.norm(cart_diff)
        return distance, cart_diff

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
                # Read structure data using the static POSCAR reader
                lattice_vectors, atom_positions, n_boron, n_carbon, total_atoms = DFTProcessor.read_POSCAR(poscar_file)
                final_energy, final_forces = self.extract_OUTCAR(outcar_file)
                
                if final_energy is None or not final_forces:
                    print(f"Could not extract data from {outcar_file}")
                    continue
                
                # Create atom types list
                atom_types = ["B"] * n_boron + ["C"] * n_carbon
                
                # Build graph for ML using enhanced PBCs
                graph = self.build_graph(atom_positions, atom_types, final_forces, final_energy, lattice_vectors)
                if graph is not None:
                    self.graphs.append(graph)
                    # Report edge count to verify PBC implementation
                    print(f"Created graph with {graph.edge_index.shape[1]} edges")
                
                # Original Excel output functionality
                ws.append(["begin"])
                for vector in lattice_vectors:
                    bohr_vector = [val * angstrom_to_bohr for val in vector]
                    ws.append(["lattice", *bohr_vector])
                
                final_forces_ha_bohr = [[val * ev_to_ha / angstrom_to_bohr for val in forces] for forces in final_forces]
                for i, position in enumerate(atom_positions, start=1):
                    bohr_position = [val * angstrom_to_bohr for val in position]
                    atom_type = atom_types[i - 1]
                    ws.append(["atom", *bohr_position, atom_type, *final_forces_ha_bohr[i - 1]])
                
                final_energy_ha = final_energy * ev_to_ha
                ws.append(["energy", final_energy_ha])
                ws.append(["charge", "0.00"])
                ws.append(["end"])
                
            except Exception as e:
                print(f"Error processing directory {root}: {e}")
                continue
        
        # Save Excel file
        output_file = os.path.join(os.path.dirname(self.dft_data_path), 'data.csv')
        wb.save(output_file)
        print(f"\nData saved to {output_file}")
        print(f"Total graphs created: {len(self.graphs)}")
        
        return self.graphs