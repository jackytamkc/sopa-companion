"""Open a crop through Cellpose's GUI module entry point (its --image_path flag selects batch mode)."""
import os
import runpy
import sys


def keep_off_apples_gpu():
    """On a Mac whose GPU fails the stain helpers' test (zarr_bridge/devices.py: PyTorch can call Apple's GPU available
    where its operations abort the process), Cellpose's window starts with "use GPU" off and runs its models on the
    CPU: its own check only moves a tensor there, which works even then. Elsewhere, and where the test passes, nothing
    changes."""
    if sys.platform != "darwin":
        return
    import torch

    from sopa_companion.zarr_bridge.devices import choose_device
    why = choose_device(torch)[2]
    if why:
        from cellpose import core
        core.use_gpu = lambda *args, **kwargs: False
        print(f"{why}. Cellpose's window runs on the CPU.", flush=True)


def main():
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 0)    # an interactive window; helpers run at nice 19
    except PermissionError:
        pass                                     # a niced parent (e.g. the gate) cannot be undone unprivileged
    except AttributeError:
        pass                                     # Windows: no niceness
    filename = sys.argv[1]
    keep_off_apples_gpu()
    from cellpose.gui import gui
    launch = gui.run
    gui.run = lambda: launch(image=filename)
    sys.argv = ["cellpose"]
    runpy.run_module("cellpose", run_name="__main__")


if __name__ == "__main__":
    main()
