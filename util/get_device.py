def get_device(device_name):
    import torch
    # Check if the device is "cpu" or "cuda"
    if device_name == "cpu":
        return torch.device("cpu")

    elif device_name == "cuda":
        # Check if CUDA is available
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA is not available on this system.")

    elif device_name == "mps":
        # Check if MPS (Metal Performance Shaders) is available (Apple Silicon)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError("MPS is not available. Make sure you're on an Apple Silicon Mac and have the right PyTorch version installed.")

    else:
        raise ValueError(f"Invalid device name: {device_name}. Please choose from 'cpu', 'cuda', or 'mps'.")
