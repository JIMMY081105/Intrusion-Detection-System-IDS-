# File:    env_setup.py
# Purpose: Shared environment configuration imported by all model scripts.
#          Detects GPU (cuML) and W&B availability, exposes dataset paths,
#          and provides a unified wandb init helper with a no-op fallback.

import os

# Resolved path to the processed dataset directory, relative to the project root.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
TARGET_COLUMN = "Target"

GPU_AVAILABLE = False
WANDB_AVAILABLE = False


def init_gpu():
    global GPU_AVAILABLE

    try:
        # cuml.accel patches sklearn transparently — no code changes needed in model scripts.
        import cuml.accel
        cuml.accel.install()

        GPU_AVAILABLE = True

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
        print("[env_setup] GPU (cuML): enabled")
        print(f"[env_setup] CUDA_VISIBLE_DEVICES: {visible_devices}")

    except Exception:
        GPU_AVAILABLE = False
        print("[env_setup] GPU (cuML): NOT available, falling back to CPU")


def init_wandb_support():
    global WANDB_AVAILABLE

    try:
        import wandb
        WANDB_AVAILABLE = True
        print("[env_setup] wandb:      enabled")
    except Exception:
        WANDB_AVAILABLE = False
        print("[env_setup] wandb:      NOT available")


def init_wandb(project, name, config=None, group=None, tags=None):
    # Returns a real wandb run if W&B is available, otherwise a silent no-op object.
    if WANDB_AVAILABLE:
        import wandb
        return wandb.init(
            project=project,
            name=name,
            config=config if config is not None else {},
            group=group,
            tags=tags if tags is not None else [],
            reinit="finish_previous"
        )
    else:
        return DummyRun()


class DummyRun:
    """Silent no-op replacement for a wandb run object."""

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# Run on import so every script that imports this file gets a consistent environment.
init_gpu()
init_wandb_support()
