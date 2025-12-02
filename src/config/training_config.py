from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """
    Central place to store training and model hyperparameters.

    This config is intentionally small for now; we can extend it
    later when the project grows (different models, datasets, etc.).
    """

    # General
    model_name: str = "gpt-mini"
    seed: int = 42

    # Optimization
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Model architecture (we will tune these later)
    vocab_size: int = 30_000
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    # Device / training details
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    log_every_n_steps: int = 50
    save_every_n_steps: int = 500
    output_dir: str = "models/checkpoints"

    def resolved_device(self) -> str:
        """
        Resolve 'auto' into a concrete device string based on availability.
        """
        import torch

        if self.device != "auto":
            return self.device

        if torch.cuda.is_available():
            return "cuda"

        # For Apple Silicon / MPS (Mac)
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        return "cpu"

    def to_dict(self) -> dict:
        """
        Small helper to convert the config into a plain dictionary
        (useful for logging, saving to JSON, etc.).
        """
        return asdict(self)