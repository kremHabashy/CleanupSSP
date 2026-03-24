# Cleanup SSP

Research code for cleaning corrupted Spatial Semantic Pointers (SSPs) with feedforward MLPs and rectified flow-matching (geodesic or Euclidean), with optional optimal-transport pairings during training.

## To run things

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

   This loads `configs/config.yaml`, ensures data under `paths.data_root`, trains each `trainer.sampling_modes` entry, saves weights, then runs evaluation.

### Weights & Biases

- Set `wandb.enabled: false` in `configs/config.yaml` to skip init from the `src` pipeline (`src/main.py` respects this).
- For `WANDB_API_KEY` and optional `wandb.entity`, see comments in the config.

## Where things live (modular layout)

| Path | Purpose |
|------|---------|
| `configs/config.yaml` | One full run (`python -m src.main`) |
| `src/main.py` | Orchestrates config, data, optional W&B, train, eval |
| `src/utils.py` | Load / validate YAML configs |
| `src/data_gen.py` | Build SSP space; ensure dataset |
| `src/train.py` | Map YAML → `TrainingManager` kwargs |
| `src/evaluate.py` | Map YAML → `EvaluationManager` |
| `cleanup_ssps/` | SSP spaces, `SSPDataset`, flow trainers, `dataset_registry`, legacy CLI |
| `utils/` | `TrainingManager`, `EvaluationManager`, W&B helpers, OT utilities |
| `trained_models/` | Checkpoints (empty in git; see below) |

## Data and checkpoints on disk

**Datasets** under `paths.data_root` use a **geometry folder** name (bundle, encoded dim, length scale, bounds), same as in `cleanup_ssps.dataset_registry.dataset_group_dirname`:

- `{group}/train/*.npy` — training targets  
- `{group}/test/*.npy` — test targets  
- `{group}/A_matrix.npy`, `{group}/dataset_meta.json`

**Checkpoints** are written under `paths.checkpoint_dir / {group}/` (the same `{group}` string as the dataset), for example:

- `feedforward.pt` (if `train_feedforward: true`)
- `drift_{sampling_mode}.pt` for each flow mode

The repo keeps an empty `trained_models/` tree via `.gitkeep`; `.pt` files stay untracked.

Training reads target `.npy` files from disk; **noise `z0`** is drawn each step in `cleanup_ssps/dataset.py` from `trainer.noise_type`. Eval uses the same noise/target types; signal-strength sweeps blend only the **model initial state** in `utils/evaluation.py`, not the dataset.

**Do not commit** a project-local `lib/python3.10/...` tree (that usually means the IDE pointed at a venv inside the repo). Use `.venv` outside the tree or a normal virtualenv; `lib/` is gitignored.

## Old multi-experiment driver

```bash
python cleanup_ssps/main.py
```

Uses `configs/experiments.yaml`. Checkpoints go under `trained_models/{dataset_group}/` the same way.

## Contact

Karim Habashy: khabashy@uwaterloo.ca
