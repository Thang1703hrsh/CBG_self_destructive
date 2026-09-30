"""Local training artifacts; no experiment-tracking account is required."""
import json
import platform
from pathlib import Path
import torch
from transformers import TrainerCallback


def hardware():
    return dict(python=platform.python_version(), torch=torch.__version__,
                gpu_count=torch.cuda.device_count(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


class JsonlTrainingLog(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(step=state.global_step, **(logs or {})), allow_nan=False)+"\n")
