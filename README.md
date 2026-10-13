# Supplementary material

> The supplementary material includes datasets and training scripts for the ARR submission (August). 

---

## Project Structure

```text
artifact/
├── satrl/                # SAT-RL framework with training scripts
├── verl/               # VeRL source code
├── data/            
├── SandboxFusion/              # Python Sandbox Backend
└── README.md
```
---

## Environment Setup
Please download the Docker image for VeRL according to https://verl.readthedocs.io/en/latest/start/install.html. Our evaluation uses https://hub.docker.com/r/vllm/vllm-openai. For Sandbox Fusion, build the Docker image according to https://github.com/bytedance/SandboxFusion.

## Dataset

data/
├── enigmata_final/                # Enigmata (For evaluation)
├── logic_sft/               # Synthetic-Logic (For SFT)
├── satbench/            # SATBench (For RL)


## Training Setup
The minimum training requirement is 4 L40S (48GB) for Qwen3.5-9B model. For Sandbox Fusion, we recommend 64 CPU cores for efficient rollout with tool calls. 

## SFT
```text
bash satrl/trainer/sft/run_all_qwen_logic_sft_verl.sh
```

## RL
Run Ray:
```text
bash satrl/sh/run_ray_container.sh
```

Run Sandbox Fusion:
```text
bash satrl/sh/run_sandbox_interactive.sh
```



Run RL:
```text
bash satrl/trainer/grpo_trainer/dapo/train.sh
```
