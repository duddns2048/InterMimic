#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run \
    --task InterMimic_Parahome \
    --cfg_env isaacgym/src/intermimic/data/cfg/parahome_train.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/parahome.yaml \
    --output checkpoints \
    --num_envs 2048 \
    --minibatch_size 16384\
    --experiment "[Intermimic_para]_sub1_clothesstand" \
    --motion_file InterAct/Parahome/s110_0_kettle_table2desk \
    --robotType parahome/sim_human/s110_intermimic.xml \
    --wandb_project intermimic \
    --headless \
    # --resume_from checkpoints/s110/nn/s110_kettle_overfitting_tmp_tmp_latest.pth \
    # --wandb_id YOUR_RUN_ID \