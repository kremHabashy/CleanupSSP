# Cleanup SSP

The goal of this project is to train models for cleaning up corrupted Spatial Semantic Pointer (SSP) representations. It includes implementations of rectified flow models and standard MLPs for reconstructing SSPs from noise. 

---

## Features

### Core Functionality
- **Data Generation**: Generate clean and corrupted SSP datasets using hexagonal tiling and other SSP spaces.
- **Models**: Implementations of Multilayer Perceptron (MLP), Rectified Flow models, and autoencoders.
- **Training and Evaluation**:
  - Train models using cosine similarity as the primary loss function.
  - Evaluate models across signal/noise strength.
  - Compare performance of different architectures and training paradigms on cleanup tasks.
- **Visualization**:
  - Generate similarity maps and contour plots to visualize SSP performance.
  - Log metrics and results using W&B integration.

### SSP-Specific Components
- **HexagonalSSPSpace**: A class that uses hexagonal tiling for SSP encoding.
- **Binding and Decoding**: Methods to bind semantic pointers, decode SSPs to their spatial representations, and clean up noisy SSPs.
- **Flexible SSP Configurations**: Support for varying SSP dimensions, length scales, and domain bounds.

---

## Project Structure

```plaintext
CleanUp/
├── cleanup_ssps/           # Μain module
│   ├── cleanup_methods/    # Rectified Flow, Feedforward, VAE and Diffusion class definitions
│   ├── dataset/            # Configuration for model dimensionalities and processing the training data
│   ├── main/               # Processing experiments
│   ├── model/              # Main architecture for the different training paradigms
│   ├── run/                # Trainer classes for the different training paradigms
│   ├── sspspace/           # Used to generate SSPs
├── power_spherical/        # Spherical distribution sampling
├── data/                   # Folder for generated datasets
│   ├── train/              
│   └── test/               
├── utils/                  # Utilities
│   ├── config_loader/      # Loading experiment configuration
│   ├── evaluation_utils/   # For calculating baseline performance (dot product max)
│   ├── evaluation/         # Evaluation functions
│   ├── generate_data/  
│   ├── training/           # Training functions
│   ├── wandb_utils/        # Logging and initialization
├── configs/  
│   ├── experiments/        # Configuration of the experiments
├── README.md               
├── requirements.txt        
```

---

## Usage

### Data Generation
Generate datasets for training and testing:

```bash
python generate_data.py
```

### Training Models
Train a rectified flow model:

```bash
python run.py
```

### Evaluation
Evaluate the trained models using the evaluation script:

```bash
python evaluation.py
```

---

## Key Configuration Parameters

### SSP Configuration
- `ssp_dim`: The dimensionality of the SSP vectors.
- `n_rotates`: Number of rotations in the hexagonal tiling.
- `n_scales`: Number of scales in the hexagonal tiling.
- `length_scale`: Scaling factor for SSP encoding.

### Training Configuration
- `batch_size`: Number of samples per training batch.
- `epochs`: Number of training epochs.
- `lr`: Learning rate for the optimizer.
- `snr`: Signal-to-Noise Ratio for dataset corruption.

---

## Contact
For questions or issues, please reach out to:
- **Karim Habashy**: khabashy@uwaterloo.ca

---
