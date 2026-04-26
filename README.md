# jupyter-classify

用于在 Jupyter 中分析视频帧注意力权重的实验仓库。当前主要 notebook 是 `spectral_analysis.ipynb`，它会读取每层保存的 PyTorch attention 张量，选择指定 layer/head 后输出诊断信息，并绘制 attention 曲线与 FFT 周期谱。

## 环境

本项目使用 `uv` 管理 Python 环境，依赖定义在 `pyproject.toml`，锁定版本在 `uv.lock`。

```bash
uv sync --dev
```

运行 Python 脚本或模块时使用：

```bash
uv run python <script-or-module>
```

## Jupyter Kernel

推荐注册名为 `jupyter-classify` 的内核：

```bash
uv run python -m ipykernel install --user --name jupyter-classify --display-name "Python (jupyter-classify)"
```

之后启动 Jupyter：

```bash
uv run jupyter lab
```

在 notebook 中选择 `Python (jupyter-classify)` 内核。

## 数据布局

已纳入仓库的样例 attention 数据位于 `attn/`：

```text
attn/
  layer0.pt
  ...
  layer29.pt
attn.zip
```

每个 `layer*.pt` 文件包含同一套张量结构：

```text
last_frame_attention_per_head: shape=(12, 72), dtype=Half
last_frame_attention_mean:     shape=(72,), dtype=Half
```

共同元数据：

```text
num_frames: 72
block_sizes: [3, 3, ..., 3]  # 共 24 个 block
extraction_method: last_frame_multi
sample_global_idx: 16
```

本地生成或额外拷贝的多次运行数据可放在类似下面的目录中：

```text
lastframe_8/
  run_000/
    layer0.pt
    ...
  run_001/
  run_002/
```

这类 `lastframe_*/` 实验输出默认被 `.gitignore` 忽略，避免把重复或临时数据提交进仓库。

## 使用 notebook

打开 `spectral_analysis.ipynb` 后，先检查第一段配置：

```python
ATTN_DIR = Path("lastframe_8/run_001")  # 本地多次运行数据
# ATTN_DIR = Path("attn")               # 已纳入仓库的样例数据
LAYER = 1
HEAD = 1
```

常用配置项：

- `ATTN_DIR`：attention 文件目录，目录下应包含 `layer{N}.pt`。
- `LAYER`：要分析的层号，当前数据为 `0` 到 `29`。
- `HEAD`：要分析的注意力头，当前每层有 `12` 个 head。
- `FFT_RANGES`：FFT 分析使用的帧范围。
- `SOFTMAX_WINDOW_START` / `SOFTMAX_WINDOW_END`：局部 softmax 的帧窗口。
- `RESPONSE_PERIOD_MIN` / `RESPONSE_PERIOD_MAX`：筛选响应周期峰值的范围。

notebook 会执行以下步骤：

1. 使用 `torch.load` 读取 `layer{LAYER}.pt`。
2. 从 `last_frame_attention_per_head` 中选择指定 head。
3. 打印 raw attention 与 softmax attention 的统计诊断。
4. 对指定帧范围做 FFT，计算周期谱峰值。
5. 绘制 attention 时间序列和周期谱图。

## Git 忽略策略

`.gitignore` 主要忽略以下内容：

- `.venv/`、`venv/` 等本地虚拟环境。
- `__pycache__/`、`.pytest_cache/` 等 Python 缓存。
- `.ipynb_checkpoints/` 等 Jupyter 自动生成文件。
- `lastframe_*/`、`outputs/`、`figures/`、`plots/`、`runs/` 等实验输出目录。
- 编辑器、系统临时文件和 `.env` 本地配置。

注意：如果某些文件已经被 Git 跟踪，后续加入 `.gitignore` 不会自动停止跟踪；需要额外执行 `git rm --cached <path>` 才能从索引中移除。
