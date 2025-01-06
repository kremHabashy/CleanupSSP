import numpy as np

def compute_cleanup_baseline(ssp_space, ssp_dim, snr, grid_resolution=32, num_trials=100):
        """
        Perform cleanup baseline by comparing corrupted SSPs to grid SSPs.

        Args:
            ssp_space: HexagonalSSPSpace object defining the SSP space.
            ssp_dim: Dimensionality of the SSPs.
            snr: Signal-to-noise ratio for corrupted SSPs.
            grid_resolution: Resolution of the grid over the environment.
            num_trials: Number of trials (random SSPs to cleanup).

        Returns:
            Mean cosine similarity for the cleanup baseline.
        """
        # Create grid of SSPs over the domain
        x_space = np.linspace(-1, 1, grid_resolution)
        y_space = np.linspace(-1, 1, grid_resolution)
        X, Y = np.meshgrid(x_space, y_space)
        grid_positions = np.vstack([X.ravel(), Y.ravel()]).T
        grid_ssps = np.array([ssp_space.encode(pos[np.newaxis, :]) for pos in grid_positions])
        grid_ssps = grid_ssps.reshape(-1, ssp_dim)

        # Sample random positions within the domain
        random_positions = np.random.uniform(-1, 1, size=(num_trials, 2))
        ground_truth_ssps = np.array([ssp_space.encode(pos[np.newaxis, :]) for pos in random_positions])
        ground_truth_ssps = ground_truth_ssps.reshape(num_trials, ssp_dim)

        # Generate corrupted SSPs
        z = np.random.randn(num_trials, ssp_dim)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        corrupted_ssps = snr * ground_truth_ssps + (1.0 - snr) * z

        # Compute similarities between corrupted SSPs and grid SSPs
        similarities = corrupted_ssps @ grid_ssps.T
        max_indices = np.argmax(similarities, axis=1)
        cleaned_up_ssps = grid_ssps[max_indices]

        # Compute cosine similarities
        dot_products = np.einsum('ij,ij->i', ground_truth_ssps, cleaned_up_ssps)
        norms_gt = np.linalg.norm(ground_truth_ssps, axis=1)
        norms_cleaned = np.linalg.norm(cleaned_up_ssps, axis=1)
        cosine_sims = dot_products / (norms_gt * norms_cleaned)

        return np.mean(cosine_sims)