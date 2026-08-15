import torch


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Priority:
        CUDA
        MPS
        CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def get_device_name() -> str:
    """
    Human-readable device description.
    """

    device = get_device()

    if device.type == "cuda":
        return torch.cuda.get_device_name(0)

    if device.type == "mps":
        return "Apple Metal Performance Shaders"

    return "CPU"