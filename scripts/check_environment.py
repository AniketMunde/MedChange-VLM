import platform
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medchange import __version__
from medchange.utils import (
    get_device,
    get_device_name,
    load_config,
)


def main() -> None:
    print("=" * 60)
    print("MedChange-VLM Environment Check")
    print("=" * 60)

    print(f"Project version : {__version__}")
    print(f"Python          : {platform.python_version()}")
    print(f"Platform        : {platform.platform()}")
    print(f"PyTorch         : {torch.__version__}")

    device = get_device()

    print(f"Device          : {device}")
    print(f"Device name     : {get_device_name()}")

    if device.type == "cuda":
        print(
            f"CUDA version    : {torch.version.cuda}"
        )

        memory_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(
            f"GPU memory      : {memory_gb:.2f} GB"
        )

    config_path = PROJECT_ROOT / "configs" / "base.yaml"

    config = load_config(config_path)

    print(
        f"Config loaded   : "
        f"{config['project']['name']}"
    )

    print("=" * 60)
    print("Environment check completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()