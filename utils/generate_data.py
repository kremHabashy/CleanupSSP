import os
import numpy as np
import torch
from cleanup_ssps.sspspace import HexagonalSSPSpace

# Function to generate and save clean SSPs in batches
def generate_and_save_clean_coordinate_ssps_batched(ssp_space, total_samples, batch_size, output_dir):
    """
    Generate and save clean SSPs in batches to handle large datasets efficiently.

    Args:
        ssp_space: An instance of HexagonalSSPSpace for generating SSPs.
        total_samples: Total number of SSPs to generate.
        batch_size: Number of SSPs to generate in each batch.
        output_dir: Directory to save the generated SSP files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    num_batches = total_samples // batch_size + (1 if total_samples % batch_size != 0 else 0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)

        # Generate SSPs for the current batch
        sample_ssps, _ = ssp_space.get_sample_pts_and_ssps(end_idx - start_idx)
        sample_ssps = torch.tensor(sample_ssps, dtype=torch.float32)

        # Save each SSP in the batch
        for i in range(end_idx - start_idx):
            clean_ssps = sample_ssps[i]
            np.save(os.path.join(output_dir, f'coordinate_ssp_{start_idx + i}.npy'), clean_ssps.numpy())

        print(f"Saved batch {batch_idx + 1}/{num_batches}")

# Define directories for storing clean data
coordinate_data_dir_train = '/Users/karimhabashy/Desktop/LOO/MASTERS/CleanUp/data/train/coordinate_ssps'
coordinate_data_dir_test = '/Users/karimhabashy/Desktop/LOO/MASTERS/CleanUp/data/test/coordinate_ssps'

# Define SSP space and related parameters
ssp_dim = 97
sides = 1
bounds = np.array([[0, sides], [0, sides]])
ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=ssp_dim, domain_bounds=bounds, length_scale=0.65)

# Split ratios and total number of samples
train_ratio = 0.8
total_samples = 10000  # Adjust for larger datasets
num_train_samples = int(train_ratio * total_samples)
num_test_samples = total_samples - num_train_samples
batch_size = 1000  # Number of SSPs per batch

# Generate and save clean SSPs for training and testing
generate_and_save_clean_coordinate_ssps_batched(
    ssp_space=ssp_space,
    total_samples=num_train_samples,
    batch_size=batch_size,
    output_dir=coordinate_data_dir_train
)

generate_and_save_clean_coordinate_ssps_batched(
    ssp_space=ssp_space,
    total_samples=num_test_samples,
    batch_size=batch_size,
    output_dir=coordinate_data_dir_test
)

print(" SSP dataset with the following configurations:")
print(f" - Train SSPs: {num_train_samples}")
print(f" - Test SSPs: {num_test_samples}")
print(f" - Domain bounds: {bounds}")
print(f" - SSP dimension: {ssp_dim}")
print(f" - Length scale: {ssp_space.length_scale}")
