# Eye Tracking Based Cognitive Evaluation of Automatic Readability Assessment Methods

Code for the paper **"Eye Tracking Based Cognitive Evaluation of Automatic Readability
Assessment Methods"**.

## Citation

(*To be completed later*)

## Installation

```bash
conda env create -f env.yaml
conda activate readability_python
```

The statistical analyses also use Julia (mixed-effects models) and R (the `cocor`
package); both are installed automatically on first run.

## Running the analyses

The full correlation + plotting pipeline runs as a tmux-based task graph. From the repo
root:

```bash
bash src/Correlations/run_wrappers/launch_tmux.sh
tmux attach -t readability
```

See [src/Correlations/run_wrappers/RUN README.md](src/Correlations/run_wrappers/RUN%20README.md)
for the task graph, the per-figure run map, and how to re-run individual tasks.

## Layout

- [src/Correlations/](src/Correlations/) — correlation analyses, statistical tests, and
  figure generation.
- [src/data/](src/data/) — readability corpora statistics.
- [src/utils/](src/utils/) — shared utilities.

Links to the resources used for calculating the readability formulas are provided in the
SI Appendix of the paper.
