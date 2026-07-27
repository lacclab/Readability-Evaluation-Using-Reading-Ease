"""Worker: poll for dep markers, run the named task, write its marker on success.

Usage:
    python -m src.Correlations.run_wrappers.run_task <task_name>
"""
import sys
import time
from pathlib import Path

from src.Correlations.run_wrappers.tasks import TASKS


MARKER_DIR = Path(__file__).parent / ".done"
MARKER_DIR.mkdir(exist_ok=True)
POLL_SECONDS = 5


def _marker(name):
    return MARKER_DIR / f"{name}.done"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python -m src.Correlations.run_wrappers.run_task <task_name>", file=sys.stderr)
        print(f"Available tasks: {list(TASKS.keys())}", file=sys.stderr)
        sys.exit(2)

    task_name = sys.argv[1]
    if task_name not in TASKS:
        print(f"Unknown task: {task_name}", file=sys.stderr)
        print(f"Available tasks: {list(TASKS.keys())}", file=sys.stderr)
        sys.exit(2)

    task = TASKS[task_name]

    if _marker(task_name).exists():
        print(f"[{task_name}] Marker already exists — task is done. Delete {_marker(task_name)} to re-run.")
        return

    deps = task["deps"]
    if deps:
        print(f"[{task_name}] Waiting for deps: {deps}", flush=True)
        while True:
            missing = [d for d in deps if not _marker(d).exists()]
            if not missing:
                break
            print(f"[{task_name}]   still waiting for: {missing}", flush=True)
            time.sleep(POLL_SECONDS)
        print(f"[{task_name}] All deps satisfied, starting.", flush=True)

    src_path = Path.cwd() / "src"
    task["fn"](src_path)

    _marker(task_name).touch()
    print(f"[{task_name}] Done. Marker written to {_marker(task_name)}.")


if __name__ == "__main__":
    main()
