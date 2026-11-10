"""The harness's detached launcher: records its environment beside the image it was given (gui-launch.json) and exits
without importing a GUI; it waits while keep-gui is there. With fail-gui there it closes with an error as a GUI
without its packages does (a line on its output first, as Cellpose prints before it downloads its logo); with kill-gui
there a signal ends it before it writes anything."""
import json
import os
from pathlib import Path
import signal
import sys
import time


if __name__ == "__main__":
    folder = Path(sys.argv[1]).parent
    (folder / "gui-launch.json").write_text(json.dumps({"image": sys.argv[1], "pid": os.getpid(),
        "session": os.getsid(0), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "display": os.environ.get("DISPLAY"), "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "cwd": os.getcwd(), "safe_path": os.environ.get("PYTHONSAFEPATH"), "log_blocking": os.get_blocking(1)}))
    if (folder / "kill-gui").exists():
        os.kill(os.getpid(), signal.SIGKILL)
    if (folder / "fail-gui").exists():
        print("downloading logo")
        print("Traceback (most recent call last):\nModuleNotFoundError: No module named 'qtpy'", file=sys.stderr)
        sys.exit(1)
    while (folder / "keep-gui").exists():
        time.sleep(.05)
