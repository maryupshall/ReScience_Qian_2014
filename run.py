"""Main run script for whole analysis.

This should be ran from the CLI as python run.py 'x'

Where x can be
    nothing to run all figures
    all to run all figures
    f1, f2, f3, f4, or f6 to run a specific figure
"""

import glob
import os
import shutil
import sys

from generators import figure_1, figure_2, figure_3, figure_4, figure_6


def run_all(function_dict):
    """Run all figures in a dict with each value being a figure module."""
    for f in function_dict.values():
        f.run()


def clean():
    """Cleanup temp files from auto."""
    try:
        shutil.rmtree("auto_temp/")
    except OSError:
        pass
    try:
        for f in glob.glob("_auto_*.so"):
            os.remove(f)
    except OSError:
        pass
    try:
        for f in glob.glob("auto_*.py"):
            os.remove(f)
    except OSError:
        pass
    try:
        for f in glob.glob("auto_*.pyc"):
            os.remove(f)
    except OSError:
        pass
    try:
        pass
        os.remove("fort.9")
    except OSError:
        pass


if __name__ == "__main__":
    switcher = {
        "f1": figure_1,
        "f2": figure_2,
        "f3": figure_3,
        "f4": figure_4,
        "f6": figure_6,
    }
    if len(sys.argv) == 2:
        if sys.argv[1] in switcher.keys():
            switcher[sys.argv[1]].run()
        else:
            run_all(switcher)
    else:
        run_all(switcher)

    clean()
