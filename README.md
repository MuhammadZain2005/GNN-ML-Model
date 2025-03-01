<h1 align="center">GNN-ML-Model</h1>

<p align="center">
    <img src="https://img.shields.io/badge/Made%20with-Python-blue.svg" alt="Made with Python">
    <img src="https://img.shields.io/badge/Made%20with-PyTorch-red.svg" alt="Made with PyTorch">
</p>

## Project Overview

This repository is part of a research project titled **"Developing a Machine Learning Model for Material Property Mapping of Advanced Ceramics for Energy and Defense Applications"**. The project focuses on leveraging machine learning techniques, particularly Graph Neural Networks (GNNs), to predict and map material properties of advanced ceramics, which are crucial in energy and defense sectors.

## Repository Structure

- **DFT_data/**: Contains Density Functional Theory (DFT) data files used for training and validating the machine learning models.

- **`DFT_processor_2_Zain.py`**: A script designed to process DFT data, extracting essential features and preparing them for input into the GNN model.

- **`ML.py`**: Implements machine learning techniques for material property prediction. It:
  - Loads and processes DFT data using `DFTProcessor`.
  - Uses `torch_geometric`'s `DataLoader` to handle structured datasets.
  - Provides functionality to train and test different GNN models (`GNNModel`, `EnhancedGNNModel`).
  - Includes visualization utilities to analyze model performance.

- **`gnn_model.py`**: Defines the Graph Neural Network architecture, including:
  - **Custom EdgeConv Layer**: A message-passing layer that incorporates edge features, improving node interactions.
  - **GNN Architectures**:
    - `GNNModel`: A base Graph Convolutional Network (GCN) using `GCNConv`.
    - `EnhancedGNNModel`: A more complex variant integrating `GINConv` and batch normalization for better feature representation.
  - **Global Pooling Mechanism**: Uses `global_mean_pool` to aggregate node features into a final representation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
