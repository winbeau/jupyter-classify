# AGENTS.md

## Environment

- This workspace is managed with `uv`.
- The project environment lives in `.venv/`.
- Install or sync dependencies with:

```bash
uv sync --dev
```

- Run Python commands inside the managed environment with:

```bash
uv run python <script-or-module>
```

## Jupyter Kernel

- The intended Jupyter kernel name is `jupyter-classify`.
- Register or refresh the kernel with:

```bash
uv run python -m ipykernel install --user --name jupyter-classify --display-name "Python (jupyter-classify)"
```

## Attention Weights

- Attention files are stored in `attn/`.
- Files present: `layer0.pt` through `layer29.pt` for 30 layers.
- Each `.pt` file is a PyTorch zip serialization containing the same tensor layout:

```text
last_frame_attention_per_head: shape=(12, 72), dtype=Half, stride=(72, 1)
last_frame_attention_mean:     shape=(72,),    dtype=Half, stride=(1,)
```

- Shared metadata observed across files:

```text
num_frames: 72
block_sizes: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
extraction_method: last_frame_multi
sample_global_idx: 16
```

- The base system `python3` did not have `torch` installed when these shapes were inspected. The shapes above were read from PyTorch serialization metadata without loading tensors through `torch`.
