# RL Public Release Checklist

Use this checklist before publishing new RL experiment updates.

## 1. Reproducibility
- Record exact commands used for training, simulation, and analysis.
- Record Python version and dependency snapshot (`pip freeze`).
- Record scenario file path and budget/step/trial settings.
- Record random seed policy (fixed or multi-seed list).

## 2. Keep in Git
- Source code in `src/` and `classes/`.
- Small configuration files and command examples.
- Aggregate outputs needed for the paper/rebuttal:
  - `performance_comparison.csv`
  - `analysis_report.md`
  - selected figures used in manuscript

## 3. Do NOT Keep in Git
- Full per-trial raw result folders.
- Training caches/checkpoints unless intentionally releasing model weights.
- Local debug logs and temporary notebooks.

## 4. Validation
- Run a short sanity run (e.g., 3-5 trials) and verify scripts complete.
- Run full experiment command once and verify output directories and files.
- Confirm analysis command reproduces table/figure values used in paper.

## 5. Paper Sync
- Update README command blocks to match current script names.
- Ensure figure/table generation paths in paper notes match repository paths.
- Add a short changelog note for major RL policy or metric changes.

