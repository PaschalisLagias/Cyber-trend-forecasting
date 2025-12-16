from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CyberVisionTSConfig:
    """Configuration dataclass for CyberVisionTS."""

    script_dir: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)

    use_gpu: bool = True
    num_workers: int = 0
    seed: int = 123

    train_split: float = 0.42
    val_split: float = 0.30
    test_split: float = 0.28

    # PCA dimensionality reduction (applied during preprocessing)
    apply_pca: bool = True
    pca_variance_ratio: float = 0.95  # Retain 95% variance

    model_arch: str = "mae_base"
    mode: str = "zero_shot"
    finetune_type: str = "ln"
    load_pretrained: bool = True

    epochs: int = 10
    batch_size: int = 1  # VisionTS expands to (batch * nvars) internally, keep small for 1231 features
    learning_rate: float = 1e-4
    patience: int = 5

    context_window_size: int = 10
    pred_window_size: int = 36
    context_len: int = 10
    pred_len: int = 36

    periodicity: int = 1
    norm_const: float = 0.4
    align_const: float = 0.4
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        self.script_dir = Path(__file__).resolve().parents[1]
        self.raw_data_path = self.script_dir / "Data_Preparation" / "Cyber_Trend_Forecasting_All.csv"
        self.output_dir = self.script_dir / "Processed_Data" / "vision"
        self.checkpoint_dir = self.output_dir / "checkpoints" / "visionts"
        self.ckpt_dir = self.output_dir / "ckpt"

    def update_from_dict(self, overrides: dict) -> None:
        """Update config attributes from a dictionary of overrides."""
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
