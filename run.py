import glob
import os
import shutil
import sys

from generators import figure_1, figure_2, figure_3, figure_4, figure_6


def run_all():
    figure_1.run()
    figure_2.run()
    figure_3.run()
    figure_4.run()
    figure_6.run()


def __clean__():
    try:
        shutil.rmtree('auto_temp/')
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
        os.remove('fort.9')
    except OSError:
        pass


if __name__ == "__main__":
    if len(sys.argv) == 2:
        switcher = {"f1": figure_1,
                    "f2": figure_2,
                    "f3": figure_3,
                    "f4": figure_4,
                    "f6": figure_6}
        if sys.argv[1] in switcher.keys():
            switcher[sys.argv[1]].run()
        else:
            run_all()
    else:
        run_all()

    __clean__()
