# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InterMimic is a physics-based whole-body human-object interaction system that trains neural policies using reinforcement learning and motion imitation. It supports SMPL-X humanoid (51 joints, 153 DOFs) and Unitree G1 robot. CVPR 2025 Highlight paper.

## Common Commands

### Environment Setup

```bash
# Isaac Gym environment (Python 3.8 required)
conda create -n intermimic-gym python=3.8
conda activate intermimic-gym
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt

# Critical: Fix shared-library lookup (run after every conda activate)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

```

### Training

```bash
# Teacher policy (single interaction)
sh isaacgym/scripts/train_teacher.sh

# High-fidelity teacher (slower, more precise)
sh isaacgym/scripts/train_teacher_new.sh

# Student policy (distillation from teachers)
sh isaacgym/scripts/train_student.sh
```

### Testing/Inference

```bash
# Teacher policy test
sh isaacgym/scripts/test_teacher.sh

# Unitree G1 robot
sh isaacgym/scripts/test_g1.sh

# Student policy
sh isaacgym/scripts/test_student.sh

# IsaacLab inference (using Isaac Gym checkpoints)
./isaaclab/scripts/test_policy.sh --checkpoint checkpoints/smplx_teachers/sub2.pth --num_envs 16
```

### Data Replay

```bash
# Isaac Gym
sh isaacgym/scripts/data_replay.sh

# IsaacLab
./isaaclab/scripts/run_data_replay.sh --num-envs 8 --motion-dir InterAct/OMOMO_new
```

## Architecture

### Core Components

**Entry Points:**
- [run.py](isaacgym/src/intermimic/run.py) - Teacher training/inference
- [run_distill.py](isaacgym/src/intermimic/run_distill.py) - Student distillation training

**Environment Layer (isaacgym/src/intermimic/env/tasks/):**
- [intermimic.py](isaacgym/src/intermimic/env/tasks/intermimic.py) - Main interaction task (observation, reward, reset logic)
- [humanoid.py](isaacgym/src/intermimic/env/tasks/humanoid.py) - SMPL-X humanoid base class
- [intermimic_all.py](isaacgym/src/intermimic/env/tasks/intermimic_all.py) - Multi-interaction variant for student training
- [intermimic_g1.py](isaacgym/src/intermimic/env/tasks/intermimic_g1.py) - Unitree G1 robot task

**Learning Layer (isaacgym/src/intermimic/learning/):**
- [intermimic_agent.py](isaacgym/src/intermimic/learning/intermimic_agent.py) - Teacher PPO agent
- [intermimic_agent_distill.py](isaacgym/src/intermimic/learning/intermimic_agent_distill.py) - Student distillation agent
- [intermimic_players.py](isaacgym/src/intermimic/learning/intermimic_players.py) - Inference player
- [network_builder.py](isaacgym/src/intermimic/learning/network_builder.py) - MLP architecture

**Reward System ([intermimic.py:880](isaacgym/src/intermimic/env/tasks/intermimic.py#L880)):**

Total reward: `reward = rb * ro * rig * rcg` (multiplicative combination)

1. **Humanoid Reward (rb)** - `compute_humanoid_reward()`:
   - `rp` (w['p']=30): Body position reward - MSE between key body positions and reference
   - `rr` (w['r']=2.5): Body rotation reward - Quaternion difference for all 52 joints
   - `rpv` (w['pv']): Body position velocity reward
   - `rrv` (w['rv']): Body rotation velocity reward
   - `energy` (w['eg1']): Energy penalty for smooth motion (penalizes joint acceleration)
   - Interaction-aware weighting: Bodies near objects have lower position weight, higher rotation weight

2. **Object Reward (ro)** - `compute_obj_reward()`:
   - `rop` (w['op']=5.0): Object position reward (in local humanoid frame)
   - `ror` (w['or']=0.1): Object rotation reward
   - `ropv` (w['opv']=0.1): Object position velocity reward
   - `rorv` (w['orv']): Object rotation velocity reward
   - `obj_energy` (w['eg2']): Object motion smoothness penalty

3. **Interaction Graph Reward (rig)** - `compute_ig_reward()`:
   - `rig` (w['ig']=5.0): Measures relative positions between key bodies and object points
   - Distance-weighted: Closer body-object pairs have higher importance
   - Captures spatial relationship between human and object

4. **Contact Graph Reward (rcg)** - `compute_cg_reward()`:
   - `rcg_hand` (w['cg_hand']=5.0): Left/right hand contact matching (indices 17-32, 36-51)
   - `rcg_other` (w['cg_other']=5.0): Other body contact matching
   - `rcg_all` (w['cg_all']=3.0): Penalizes unexpected contacts
   - `contact_energy` (w['eg3']): Contact force smoothness penalty

**Reset Conditions:**
- `human_reset`: Mean key body position error > 0.5m
- `object_reset`: Mean object point cloud error > 0.5m
- `ig_reset`: Interaction graph relative error > 2x
- `contact_reset`: Hand contact mismatch accumulation

**IsaacLab Integration (isaaclab/src/intermimic_lab/):**
- [intermimic_env.py](isaaclab/src/intermimic_lab/intermimic_env.py) - DirectRLEnv implementation
- [policy_loader.py](isaaclab/src/intermimic_lab/policy_loader.py) - Loads Isaac Gym checkpoints for IsaacLab

### Configuration System

Environment configs: `isaacgym/src/intermimic/data/cfg/omomo_*.yaml`
Training configs: `isaacgym/src/intermimic/data/cfg/train/rlg/*.yaml`

Key parameters:
- `numEnvs`: Parallel environments (default: 2048, reduce to 4-16 for testing)
- `episodeLength`: Episode length (default: 300)
- `stateInit`: Initialization mode (Start/Random/Hybrid)
- `physicalBufferSize`: PSI buffer size (default: 3, set >1 to enable PSI)
- `rewardWeights`: Multi-task reward weights (p=pose, r=rotation, op/or=object, cg=contact)

### Data Flow

```
Motion Capture (.npz) → Load Motion → Observation Extraction → Policy Network → Actions → PD Control → Simulation
```

### Code Execution Flow (train_teacher.sh --task InterMimic)

```
train_teacher.sh
└── python -m intermimic.run
    └── main() [run.py:198]
        ├── get_args() → parse CLI arguments
        ├── load_cfg() → load omomo_train.yaml, omomo.yaml
        ├── wandb.init() (if --wandb)
        ├── build_alg_runner() [run.py:188]
        │   └── Register: intermimic_agent, intermimic_players, intermimic_models, intermimic_network_builder
        ├── runner.load(cfg_train)
        ├── runner.reset()
        └── runner.run(vargs)
            └── InterMimicAgent.train() [intermimic_agent.py:58]
                └── CommonAgent.train() [common_agent.py:104]
                    ├── init_tensors()
                    ├── env_reset() → InterMimic.reset()
                    └── while True: (Training Loop)
                        ├── train_epoch() [intermimic_agent.py:259]
                        │   ├── play_steps() [intermimic_agent.py:108] (Rollout Collection)
                        │   │   └── for n in horizon_length:
                        │   │       ├── get_action_values() → model forward pass
                        │   │       └── env_step(actions)
                        │   │           └── RLGPUEnv.step() [run.py:146]
                        │   │               └── InterMimic.step() [base_task.py:135]
                        │   │                   ├── pre_physics_step() [humanoid.py:387]
                        │   │                   │   └── PD control → set_dof_position_target_tensor()
                        │   │                   ├── _physics_step() → gym.simulate()
                        │   │                   └── post_physics_step() [humanoid.py:400]
                        │   │                       ├── _refresh_sim_tensors()
                        │   │                       ├── _compute_hoi_observations()
                        │   │                       ├── _compute_observations()
                        │   │                       ├── _compute_reward() [intermimic.py:880]
                        │   │                       │   ├── compute_humanoid_reward() → rb
                        │   │                       │   ├── compute_obj_reward() → ro
                        │   │                       │   ├── compute_ig_reward() → rig
                        │   │                       │   └── compute_cg_reward() → rcg
                        │   │                       │   └── rew_buf = rb * ro * rig * rcg
                        │   │                       └── _compute_reset()
                        │   ├── prepare_dataset() → batch for training
                        │   └── for mini_epoch: train_actor_critic()
                        │       └── calc_gradients() [intermimic_agent.py:330]
                        │           ├── model forward → actor/critic loss
                        │           └── optimizer step
                        ├── _log_train_info() → TensorBoard logging
                        ├── print stats [common_agent.py:159]
                        │   └── "epoch_num:{} mean_rewards:{} fps step: {} fps total: {}"
                        └── save() (if save_freq)
```

**Key Classes Inheritance:**
```
InterMimic (intermimic.py)
└── Humanoid_SMPLX (humanoid.py)
    └── BaseTask (base_task.py)

InterMimicAgent (intermimic_agent.py)
└── CommonAgent (common_agent.py)
    └── A2CAgent (rl_games)
        └── A2CBase (rl_games)
```

## Key Technical Details

- All scripts set PYTHONPATH to `$REPO_ROOT/isaacgym/src:$REPO_ROOT`
- Motion data stored in `InterAct/OMOMO_new/` as `.npz` files
- Checkpoints stored in `checkpoints/` directory
- Physics runs at 60Hz (controlFrequencyInv=2 means 30Hz control)
- Early termination at height 0.15m
- RL framework: rl_games 1.1.4 (PPO)

## Debugging

Common issues:
- Isaac Gym import fails: Set `LD_LIBRARY_PATH` as shown above
- PYTHONPATH errors: Ensure `isaacgym/src` and repo root are in path
- Memory issues: Reduce `numEnvs` in config
- Checkpoint loading: Use paths relative to repo root or absolute paths