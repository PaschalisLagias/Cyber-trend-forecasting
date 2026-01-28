from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CyberVisionTSppConfig:
    """Configuration dataclass for CyberVisionTS++ (multivariate extension)."""

    script_dir: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)

    use_gpu: bool = True
    num_workers: int = 0
    seed: int = 123

    # Data splits - same as VisionTS for consistency
    train_split: float = 0.43
    val_split: float = 0.30
    test_split: float = 0.27

    # Feature selection (VisionTS++ works better with selected features)
    # Max 224 features due to shared image space (1 pixel per feature at max)
    use_feature_selection: bool = True
    n_features: int = 224

    # Model architecture
    model_arch: str = "mae_base"
    mode: str = "train"
    finetune_type: str = "full"
    load_pretrained: bool = True

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 50
    temporal_weight: float = 0.3  # Weight for temporal difference loss component
    corr_weight: float = 1.0  # Weight for correlation loss component (higher to prioritize pattern matching)

    # Sequence lengths
    context_window_size: int = 10
    pred_window_size: int = 36
    context_len: int = 10
    pred_len: int = 36

    # VisionTS++ specific parameters
    periodicity: int = 12
    norm_const: float = 0.4
    align_const: float = 0.4
    interpolation: str = "bilinear"
    padding_mode: str = "replicate"

    # VisionTS++ multivariate options
    quantile: bool = True  # Enable quantile predictions for uncertainty (required for pretrained weights)
    quantile_head_num: int = 9  # Number of quantile heads
    color: bool = True  # Use color channels to encode variables
    clip_input: int = 0  # Input clipping mode (0 = standard)
    complete_no_clip: bool = False  # Disable all clipping
    num_patch_input: Optional[int] = None  # Manual control of input patches (None = auto)

    def __post_init__(self) -> None:
        self.script_dir = Path(__file__).resolve().parents[1]
        self.raw_data_path = self.script_dir / "Data_Preparation" / "Cyber_Trend_Forecasting_All.csv"
        self.output_dir = self.script_dir / "Processed_Data" / "visionpp"
        self.checkpoint_dir = self.output_dir / "checkpoints" / "visiontspp"
        self.ckpt_dir = self.output_dir / "ckpt"

    def update_from_dict(self, overrides: dict) -> None:
        """Update config attributes from a dictionary of overrides."""
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
