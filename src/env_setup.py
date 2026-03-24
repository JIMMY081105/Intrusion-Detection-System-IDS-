#this file is used to setup the environment for all scripts
#it checks whether gpu and wandb are available
#it also stores shared dataset path and target column
#make single-gpu runs explicit and reproducible

#import os for environment variables and path handling
import os

#basic dataset settings shared by all models
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
TARGET_COLUMN = "Target"

#global flags
GPU_AVAILABLE = False
WANDB_AVAILABLE = False


#define gpu initialization
def init_gpu():
    global GPU_AVAILABLE

    try:
        #install cuml acceleration hook for sklearn-compatible gpu acceleration
        import cuml.accel
        cuml.accel.install()

        GPU_AVAILABLE = True

        #(show which gpu this process can see for reproducibility and auditability)
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")

        print("[env_setup] GPU (cuML): enabled")
        print(f"[env_setup] CUDA_VISIBLE_DEVICES: {visible_devices}")

    except Exception:
        GPU_AVAILABLE = False
        print("[env_setup] GPU (cuML): NOT available, falling back to CPU")


#define wandb initialization check
def init_wandb_support():
    global WANDB_AVAILABLE

    try:
        import wandb
        WANDB_AVAILABLE = True
        print("[env_setup] wandb:      enabled")
    except Exception:
        WANDB_AVAILABLE = False
        print("[env_setup] wandb:      NOT available")


#define shared wandb init function
def init_wandb(project, name, config=None, group=None, tags=None):
    #(returns a real wandb run if available, otherwise a dummy run object)
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


#dummy run so scripts can still work even if wandb is unavailable
class DummyRun:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


#run setup immediately when this file is imported
init_gpu()
init_wandb_support()