#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/grab_train.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/grab.yaml \
    --output checkpoints \
    --experiment "[Intermimic]_grab_s1_apple,table_10_334" \
    --num_envs 2048 \
    --minibatch_size 16384\
    --motion_file InterAct/GRAB/s1_apple,table_10_334 \
    --wandb_project intermimic \
    --headless \
    # --resume_from checkpoints/s110/nn/s110_kettle_overfitting_tmp_tmp_latest.pth \
    # --wandb_id YOUR_RUN_ID \