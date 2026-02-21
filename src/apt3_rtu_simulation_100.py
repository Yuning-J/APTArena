#!/usr/bin/env python3
"""
Deprecated compatibility wrapper.

Use `src/apt3_simulation_main.py` for simulation runs.
"""

import sys

from apt3_simulation_main import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] `src/apt3_rtu_simulation_100.py` is a compatibility wrapper. "
        "Please use `src/apt3_simulation_main.py`.",
        file=sys.stderr,
    )
    main()
