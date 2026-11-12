"""The harness's detached launcher: records its environment beside the image it was given (gui-launch.json) and exits
without importing a GUI; it waits while keep-gui is there. With fail-gui there it closes with an error as a GUI
without its packages does (a line on its output first, as Cellpose prints before it downloads its logo); with kill-gui
there a signal ends it before it writes anything. With qt-libraries-gui or qt-display-gui there it prints what Qt
(PyQt6 6.10) prints when it can open no window, for want of the system's libraries or of a display, and closes with an
error."""
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
    qt = ['qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.',
          'qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.',
          "This application failed to start because no Qt platform plugin could be initialized. Reinstalling the "
          "application may fix this problem.", "",
          "Available platform plugins are: wayland-egl, wayland, xcb, linuxfb, minimalegl, vnc, minimal, eglfs, offscreen.", ""]
    for flag, first in (("qt-libraries-gui", []), ("qt-display-gui", ["qt.qpa.xcb: could not connect to display "])):
        if (folder / flag).exists():
            print("\n".join([*first, *qt]), file=sys.stderr)
            sys.exit(134)
    while (folder / "keep-gui").exists():
        time.sleep(.05)
