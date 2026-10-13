# SAT-RL

Code, datasets, and training scripts for **SAT-RL: Training Language Models for Logical Reasoning with Symbolic Solvers** by Jialiang Sun and Kuldeep Meel. SAT-RL combines supervised warm-up on synthetic solver traces with solver-grounded reinforcement learning. The paper trains Qwen3.5-4B and Qwen3.5-9B using DAPO.

## Project Structure

```text
SAT-RL/
├── satrl/          # Training scripts and configurations
├── verl/           # Bundled verl source with SAT-RL tools and rewards
├── data/           # Synthetic-Logic, SAT-Bench, and Enigmata datasets
├── SandboxFusion/  # Code execution backend
└── README.md
```

## Environment Setup

The paper reports the following software versions:

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| CUDA Toolkit | 12.9 |
| PyTorch | 2.11.0 |
| verl | 0.8.0 |
| vLLM | 0.19.1 |
| Transformers | 5.5.4 |

Prepare the training environment using the [bundled verl installation guide](verl/docs/start/install.rst) and the sandbox image using the [SandboxFusion setup instructions](SandboxFusion/README.md#contributing). Match the training environment to the versions above; the bundled guide also contains older environment examples.

From the root of this checkout, record its absolute path:

```bash
export SATRL_REPO="$(pwd)"
```

Inside the training environment, install the bundled verl source and the solver dependencies used by the logic-equivalence reward:

```bash
python -m pip install --no-deps -e "$SATRL_REPO/verl"
python -m pip install z3-solver pysmt python-sat ortools sympy
```

Install the symbolic solver packages in the SandboxFusion image's Python execution environment before using it:

```bash
conda run -n sandbox-runtime python -m pip install \
  z3-solver pysmt python-sat ortools sympy
```

The host Python used by the SandboxFusion launcher also needs PyYAML:

```bash
python -m pip install PyYAML
```

## Datasets

```text
data/
├── enigmata_final/       # Enigmata, for out-of-domain evaluation
│   └── tools_symbolic/
│       ├── train.parquet
│       └── test.parquet
├── logic_sft/           # Synthetic-Logic, for supervised warm-up
│   ├── train.parquet
│   └── val.parquet
└── satbench/            # SAT-Bench, for RL and in-domain evaluation
    ├── SATBench-problems.jsonl
    └── tools_symbolic/
        ├── train.parquet
        └── test.parquet
```

## Training Setup

The paper's distributed training setup uses two nodes, each with four L40S GPUs (48 GB each) and 64 CPU cores. SFT uses one epoch; RL uses two epochs, DAPO with eight rollouts per prompt, and prompt/response limits of 4,096/8,192 tokens.

The Ray, SandboxFusion, and RL launchers are Slurm/Apptainer scripts. They expect `PROJECT` to identify the cluster project directory, `SCRATCH` to identify checkpoint storage, and `$PROJECT/logic` to be the root of this checkout. For a checkout elsewhere, adapt these paths before launching:

| File | Paths to configure |
| --- | --- |
| [Ray launcher](satrl/sh/run_ray_container.sh) | Training image at `$PROJECT/sandbox/verl.sif`; shared Ray parameters under `$PROJECT/logic/output/params` |
| [Sandbox launcher](satrl/sh/run_sandbox_interactive.sh) | `SF_DIR` must point to this checkout's `SandboxFusion/`; shared tool configuration under `$PROJECT/logic/output/params` |
| [Sandbox runner](SandboxFusion/run_no_background.py) | Server image at `$PROJECT/sandbox/code_sandbox_server.sif`; shared tool configuration directory |
| [RL launcher](satrl/trainer/grpo_trainer/train.sh) | `CONFIG_PATH`, `TRAIN_DIR`, `VAL_DIR`, and `custom_reward_function.path` must point to this checkout's configurations, datasets, and reward code; shared tool configuration directory |

Keep the generated `output/params` paths consistent across these launchers, and make the checkout, images, and checkpoint storage accessible inside the containers on every allocated node.

## SFT

Run inside the training environment. Select the paper's models and the supplied Synthetic-Logic files explicitly:

```bash
TRAIN_DATA="$SATRL_REPO/data/logic_sft/train.parquet" \
EVAL_DATA="$SATRL_REPO/data/logic_sft/val.parquet" \
bash "$SATRL_REPO/satrl/trainer/sft/run_all_qwen_logic_sft_verl.sh" \
  --models "qwen3.5-4b qwen3.5-9b"
```

The models train sequentially. Add `DRY_RUN=1` before `TRAIN_DATA=...` to inspect the commands. Export the resulting SFT checkpoint to a Hugging Face model directory before RL; the [checkpoint conversion launcher](satrl/trainer/sft/convert_latest_verl_sft_checkpoints_to_hf.sh) documents its options with `--help`.

## RL

Within a Slurm allocation, start Ray and SandboxFusion from the host after configuring the paths above:

```bash
bash "$SATRL_REPO/satrl/sh/run_ray_container.sh"
bash "$SATRL_REPO/satrl/sh/run_sandbox_interactive.sh"
```

Then, inside the training container for the same job, load the Ray parameters and run RL. Replace the model path with the exported SFT model directory. Run from `verl/` so Hydra can resolve the bundled trainer configurations.

```bash
source "$PROJECT/logic/output/params/ray_params_${SLURM_JOB_ID}.env"
cd "$SATRL_REPO/verl"
bash "$SATRL_REPO/satrl/trainer/grpo_trainer/train.sh" \
  --mode tools_symbolic \
  --dataset satbench \
  --model /absolute/path/to/Qwen3.5-9B-SFT
```
