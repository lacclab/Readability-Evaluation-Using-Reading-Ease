#!/usr/bin/env bash
# Launch the full correlation pipeline in a tmux session, one window per task.
# Each task waits for its dependencies via marker files at .done/<task>.done.
#
# Usage:
#   bash src/Correlations/run_wrappers/launch_tmux.sh
#   tmux attach -t readability

set -euo pipefail

SESSION="readability"
REPO_ROOT="$(git rev-parse --show-toplevel)"

TASKS=(
    # C0_prepare_data is intentionally OFF by default — it regenerates the underlying data
    # (aligned eye df -> eye metrics -> reading speed -> reading comprehension -> item
    # difficulty) and is only needed when the source export changes. To regenerate, run it
    # to completion first, then launch this pipeline:
    #   python -m src.Correlations.run_wrappers.run_task C0_prepare_data
    C1_calc_l1_l2_first
    C2_calc_l1_l2_gath_hunt
    C3_calc_l1_first
    C4_calc_l2_first
    C5_calc_pair_plots
    C6_calc_perm_tests
    P1_plot_main
    P2_plot_sm_l1_l2
    P3_plot_sm_hunting
    P4_plot_within_metrics
    P5_plot_perm_tests
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Kill it first with: tmux kill-session -t $SESSION"
    exit 1
fi

# create session with a placeholder window; we'll delete it at the end
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" -n "_launcher"

for task in "${TASKS[@]}"; do
    tmux new-window -t "$SESSION" -n "$task" -c "$REPO_ROOT"
    tmux send-keys -t "$SESSION:$task" \
        "conda activate readability_python && python -m src.Correlations.run_wrappers.run_task $task" C-m
done

tmux kill-window -t "$SESSION:_launcher" 2>/dev/null || true

echo "Session '$SESSION' created with ${#TASKS[@]} windows."
echo "Attach with:  tmux attach -t $SESSION"
echo "Kill with:    tmux kill-session -t $SESSION"
