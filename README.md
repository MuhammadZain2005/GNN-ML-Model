 <h1 align="center">GNN-ML-Model</h1>

<p align="center">
    <img src="https://img.shields.io/badge/Made%20with-Python-blue.svg" alt="Made with Python">
    <img src="https://img.shields.io/badge/Made%20with-PyTorch-red.svg" alt="Made with PyTorch">
</p>

## Project Overview

This repository is part of a research project titled **"Developing a Machine Learning Model for Material Property Mapping of Advanced Ceramics for Energy and Defense Applications"**. The project focuses on leveraging machine learning techniques, particularly Graph Neural Networks (GNNs), to predict and map material properties of advanced ceramics, which are crucial in energy and defense sectors.

## Repository Structure

- **DFT_data/**: Contains Density Functional Theory (DFT) data files used for training and validating the machine learning models.

- **DFT_processor_2_Zain.py**: A script designed to process DFT data, extracting essential features and preparing them for input into the GNN model.

- **ML.py**: Implements traditional machine learning algorithms to analyze and predict material properties based on the processed DFT data.

- **gnn_model.py**: Defines and trains the Graph Neural Network model, aiming to capture complex relationships in the material structures to enhance prediction accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
