#!/usr/bin/env bash
set -euo pipefail

HBC_LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_LAB_ROOT="${ROBOT_LAB_ROOT:-$(cd "${HBC_LAB_ROOT}/.." && pwd)/robot_lab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${HBC_LAB_ROOT}/.." && pwd)/IsaacLab}"

export PYTHONPATH="${HBC_LAB_ROOT}/source/hbc_lab:${ROBOT_LAB_ROOT}/source/robot_lab:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_mimic:${ISAACLAB_ROOT}/source/isaaclab_rl:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage: ./hbc_lab.sh [command] [args...]

Commands:
  -t, train      Launch HBC RSL-RL training.
  -p, play       Launch HBC RSL-RL play.
  -l, list-envs  List HBC gym environment ids.
  -i, install    Install hbc_lab in editable mode.
EOF
}

case "${1:-}" in
  -t|train)
    shift
    python "${HBC_LAB_ROOT}/scripts/rsl_rl/train.py" "$@"
    ;;
  -p|play)
    shift
    python "${HBC_LAB_ROOT}/scripts/rsl_rl/play.py" "$@"
    ;;
  -l|list-envs)
    shift || true
    python "${HBC_LAB_ROOT}/scripts/tools/list_envs.py" "$@"
    ;;
  -i|install)
    shift || true
    python -m pip install -e "${HBC_LAB_ROOT}"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
