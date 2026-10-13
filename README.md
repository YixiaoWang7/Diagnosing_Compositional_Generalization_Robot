# Diagnosing Compositional Generalization in Sequential Robot Tasks

> 🌟 **Accepted to CoRL 2026 · Spotlight Presentation**

Yixiao Wang, Cheng-En Wu, Lingfeng Sun, Pengcheng Wang, Xiang Ji,
Boyuan Liang, Guojian Zhan, and Masayoshi Tomizuka

[Paper](https://arxiv.org/abs/2607.29687) ·
[PDF](https://arxiv.org/pdf/2607.29687) ·
[Project page](https://yixiaowang7.github.io/Diagnosing_Compositional_Generalization_Robot_Page/) ·
[Code](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot) ·
[Datasets](#datasets-and-checkpoints) ·
[Experiment guide](projects/corl2026_cg/README.md)

We study how instruction coverage affects generalization to unseen combinations
in sequential robot manipulation. This repository provides the RoboPolicy learning
library and the paper's data preparation, training, and evaluation tools.

## Tasks and baseline results

| Task | Instruction | Success rate |
| --- | --- | ---: |
| Pick-and-Place (PP) | Object and destination | 77.75% |
| Pick-Place-Press (PPP) | Object, destination, then button | 71.31% |
| Dependent Two-Stage Pick-and-Place (2S-PP) | First object, second object, and second destination | 45.58% |
| Full-instruction two-stage baseline | Both object/destination pairs | 62.44% |

This release uses an improved training pipeline, a refactored policy, and a more
efficient data loader with lower memory use and faster video decoding. Results
may differ from those in the paper. See the
[validation report](docs/validation/cg_training_evaluation_setups_20261001.md) for details.

## Setup

Use Linux, Python 3.10, an NVIDIA GPU compatible with CUDA 12.8, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Run the commands below from the repository root after cloning.

### 1. Install

```bash
git clone https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot.git
cd Diagnosing_Compositional_Generalization_Robot
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -c constraints/training.txt \
  -e '.[dataset,opencv,vision,logging]' -e 'projects/corl2026_cg[video]'
.venv/bin/cg --help
```

If `.venv` already exists, reuse it instead of creating another environment.

### 2. Install the simulator

Simulation uses [robot_envs](https://github.com/YixiaoWang7/robot_envs) at the pinned revision:

```bash
mkdir -p repos
git clone https://github.com/YixiaoWang7/robot_envs.git repos/robot_envs
git -C repos/robot_envs checkout c7e06a0784853d74db57ebf78d7a66fa40c2df99
uv pip install --python .venv/bin/python -c constraints/training.txt \
  -e repos/robot_envs/CG/robosuite -e repos/robot_envs/CG/mimicgen \
  'mujoco==3.3.7' 'numpy>=1.26,<3' 'opencv-python>=4.11,<4.12' \
  'imageio>=2.37.3' 'imageio-ffmpeg>=0.6'
export MUJOCO_GL=egl
```

Keep the pinned simulator checkout unmodified. For rendering setup and smoke tests,
see the [simulator guide](projects/corl2026_cg/docs/simulator.md).

### 3. Set up experiment tracking

```bash
.venv/bin/wandb login
```

Training and evaluation use separate W&B runs. Replace `YOUR_ENTITY` below with
your W&B username or team. See [logging options](docs/training.md) for checkpoint
uploads and local-only runs.

## Datasets and checkpoints

| Task | Dataset | Prepared directory |
| --- | --- | --- |
| PP | [yixiiiiii/cg_l4_1s](https://huggingface.co/datasets/yixiiiiii/cg_l4_1s) | `l4` |
| PPP | [yixiiiiii/cg_l5](https://huggingface.co/datasets/yixiiiiii/cg_l5) | `l5` |
| Two-stage | [yixiiiiii/cg_l4_2s](https://huggingface.co/datasets/yixiiiiii/cg_l4_2s) | `l4_2s` |

Both two-stage variants use the same dataset. To download and prepare PP:

```bash
.venv/bin/hf download yixiiiiii/cg_l4_1s --repo-type dataset \
  --local-dir data/source/l4
.venv/bin/cg prepare --source data/source/l4 --destination data/corl2026_cg/l4
```

For another task, replace the dataset name and `l4` directory suffixes using the
table above. See the [data preparation guide](projects/corl2026_cg/README.md#prepare-data-once)
for details. Pretrained checkpoint downloads are not yet available.

## Train and evaluate

This example runs PP. Other tasks are listed in the
[configuration index](projects/corl2026_cg/configs/README.md#release-baselines).

```bash
.venv/bin/cg generate --config projects/corl2026_cg/configs/release_pp_50k.json \
  --set run.logging.wandb.entity=YOUR_ENTITY \
  --set evaluation.logging.wandb.entity=YOUR_ENTITY
MUJOCO_GL=egl .venv/bin/cg run \
  --jobs runs/corl2026_cg/release/release_pp_50k_*/experiment.json
.venv/bin/cg report --jobs runs/corl2026_cg/release/release_pp_50k_* \
  --output runs/corl2026_cg/reports/release_pp_50k
```

Configs default to `data/corl2026_cg/<family>` and `repos/robot_envs`. Override
paths with `--set run.dataset.root=...` and `--set evaluation.simulator_root=...`.
Use `--dry-run` to inspect a job; if several jobs match the glob, use its exact path.

Checkpoints are saved to `train/latest.pt` inside the job directory; metrics and
videos go to `evaluation/`. Use `cg run --skip-eval` for training only or
`--eval-only` for a trained job. See the
[experiment guide](projects/corl2026_cg/README.md#resolve-train-evaluate-report)
for resume and fine-tuning.

## Repository structure

| Location | Contents |
| --- | --- |
| `src/robopolicy/` | Policy, data loading, training, and checkpoint library |
| `projects/corl2026_cg/` | Paper experiments, task definitions, configs, and evaluation |
| `scripts/policies/`, `examples/` | Library entry points and examples |
| `docs/` | Technical guides and validation reports |

## Citation

```bibtex
@inproceedings{wang2026diagnosing,
  title={Diagnosing Compositional Generalization in Sequential Robot Tasks},
  author={Yixiao Wang and Cheng-En Wu and Lingfeng Sun and Pengcheng Wang and
          Xiang Ji and Boyuan Liang and Guojian Zhan and Masayoshi Tomizuka},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026},
  note={Spotlight presentation},
  url={https://arxiv.org/abs/2607.29687}
}
```
