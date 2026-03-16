from utils.evaluation_utils import compute_cleanup_baseline
from cleanup_ssps.sspspace import HexagonalSSPSpace
import numpy as np
import os
import plotly.graph_objects as go

signal_strengths = (1 - np.linspace(0, 1, 25)).tolist()
grid_resolutions = [16, 32, 64, 128]
methods = ['grid', 'sobol', 'Rd']

ssp_dim = 13*13*6 + 1
ssp_space = HexagonalSSPSpace(
    domain_dim=2,
    ssp_dim=ssp_dim,
    domain_bounds=np.array([[2, 1+2], [2, 1+2]]),
    length_scale=0.2,
    n_rotates=13,
    n_scales=13
)

results = {}
std_devs = {}

for method in methods:
    results[method] = {}
    std_devs[method] = {}
    for resolution in grid_resolutions:
        print(f"Method: {method}, Resolution: {resolution}")
        results[method][resolution] = []
        std_devs[method][resolution] = []
        for signal_strength in signal_strengths:
            mean_cosine_sim, std_cosine_sim = compute_cleanup_baseline(ssp_space, ssp_dim, signal_strength,
                                                         grid_resolution=resolution, method=method, num_trials=10)
            results[method][resolution].append(mean_cosine_sim)
            std_devs[method][resolution].append(std_cosine_sim)

# Create an interactive Plotly figure
fig = go.Figure()

for method in methods:
    for resolution in grid_resolutions:
        fig.add_trace(go.Scatter(
            x=signal_strengths,
            y=results[method][resolution],
            mode='lines',
            name=f'{method} - {resolution}',
            error_y=dict(
                type='data',
                array=std_devs[method][resolution],
                visible=True
            )
        ))

fig.update_layout(
    title="Average Cosine Similarity vs Signal Strength",
    xaxis_title="Signal Strength",
    yaxis_title="Average Cosine Similarity"
)

# Save the interactive plot as an HTML file
output_dir = '/u1/khabashy/CleanupSSP/results'
os.makedirs(output_dir, exist_ok=True)
html_output_file = os.path.join(output_dir, 'interactive_plot.html')
fig.write_html(html_output_file)
print(f"Interactive plot saved to {html_output_file}")
