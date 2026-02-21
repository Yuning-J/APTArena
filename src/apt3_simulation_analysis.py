"""
Unified analysis entry point for APT3 simulation outputs.

It dispatches to existing analysis scripts without changing their behavior.
"""

import argparse
import os
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analysis for APT3 simulation outputs.")
    parser.add_argument(
        "--mode",
        choices=["standard", "systematic"],
        default="standard",
        help="Analysis backend to execute.",
    )
    parser.add_argument(
        "analysis_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected analysis script.",
    )
    args = parser.parse_args()

    script_map = {
        "standard": "analyze_simulation_results.py",
        "systematic": "simulation_analysis_systematic.py",
            }
    script_name = script_map[args.mode]
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    forwarded = list(args.analysis_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    old_argv = sys.argv
    try:
        sys.argv = [script_name] + forwarded
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

