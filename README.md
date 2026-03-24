# Cleanup SSP

Research code for cleaning corrupted Spatial Semantic Pointers (SSPs) with feedforward MLPs and rectified flow-matching (geodesic or Euclidean), with optional optimal-transport pairings during training.

## Quick start (recommended)

1. **Python 3.10+**, then from the repo root:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

   Install a CUDA build of PyTorch if you use `trainer.device: cuda` in the config.

2. **Edit** `configs/config.yaml` (paths, SSP geometry, data sizes, trainer, eval, W&B).

3. **Run**:

   ```bash
   python -m src.main
   ```

   This loads `configs/config.yaml`, ensures data under `paths.data_root`, trains each `trainer.sampling_modes` entry, then runs evaluation.

### Weights & Biases

- Set `wandb.enabled: false` in `configs/config.yaml` to skip init from the `src` pipeline (`src/main.py` respects this).
- For `WANDB_API_KEY` and optional `wandb.entity`, see comments in the config.

## Where things live (modular layout)

| Path | Purpose |
|------|---------|
| `configs/config.yaml` | **Single source of truth** for one full run (`python -m src.main`) |
| `src/main.py` | Orchestrates config, data, optional W&B, train, eval |
| `src/utils.py` | Load / validate YAML, resolve relative paths |
| `src/data_gen.py` | Build SSP space; call `ensure_target_dataset` |
| `src/train.py` | Map YAML → `TrainingManager` kwargs |
| `src/evaluate.py` | Map YAML → `EvaluationManager` |
| `cleanup_ssps/` | SSP spaces, `SSPDataset`, flow trainers, `dataset_registry`, legacy CLI |
| `utils/` | `TrainingManager`, `EvaluationManager`, W&B helpers, OT utilities |
| `tests/` | `python -m unittest discover -s tests` |

## Data on disk

Under `paths.data_root`, datasets use a **geometry folder** (bundle, encoded dim, length scale, bounds) with flat splits:

- `{group}/train/*.npy` — training targets  
- `{group}/test/*.npy` — test targets  
- `{group}/A_matrix.npy` — axis matrix for that run  
- `{group}/dataset_meta.json` — hash / counts / paths  

Older trees (`{group}/dataset_{hash}/…` or `dataset_{hash}/` at root) are still detected. Set `data.train_subdir` / `data.test_subdir` in YAML if your folders use a different layout (e.g. legacy `train/targets`).

Training reads target `.npy` files from disk; **noise `z0`** is drawn each step in `cleanup_ssps/dataset.py` from `trainer.noise_type` (hypersphere or Gaussian). Eval uses the same noise/target types; signal-strength sweeps blend noise and target only when building the **model initial state** (see `utils/evaluation.py`), not inside the dataset.

**Windows:** keep `trainer.dataloader_num_workers: 0` unless you are sure multiprocessing DataLoader helps.

## Legacy multi-experiment driver

```bash
python cleanup_ssps/main.py
```

Uses `configs/experiments.yaml` (list under `experiments:`). Prefer `src.main` + `config.yaml` for new work.

## Submodule (optional)

`.gitmodules` references `power_spherical`; **`pip install -r requirements.txt`** already pulls `power-spherical` from PyPI, so you do not need the submodule for a normal install.

## Tests

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

## Contact

Karim Habashy: khabashy@uwaterloo.ca
