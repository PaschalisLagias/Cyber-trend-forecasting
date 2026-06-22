from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DATASETS = {
    "cyber_trend": {
        "raw_relpath": "Data_Preparation/Cyber_Trend_Forecasting_All.csv",
        "subdir": "cyber_trend",
        "defaults": {},
    },
    "csis": {
        "raw_relpath": "Data_Preparation/CSIS/csis_output_20260404-01.csv",
        "subdir": "csis",
        "defaults": {
            "finetune_type": "attn",
            "patience": 15,
            "mse_weight": 1.0,
        },
    },
    "mark3": {
        "raw_relpath": "Processed_Data/VisionTS/Mark3_Clipped_Data.csv",
        "subdir": "mark3",
        "defaults": {
            "finetune_type": "attn",
            "patience": 15,
        },
    },
    "v2_1": {
        "raw_relpath": "Data_Preparation/Cyber_Trend_Forecasting_All_v2_1.csv",
        "subdir": "v2_1",
        "defaults": {},
    },
}


@dataclass
class CyberVisionTSppConfig:
    """Configuration dataclass for CyberVisionTS++."""

    _MODULE_DEFAULTS: dict = field(init=False, repr=False, default_factory=lambda: {
        "finetune_type": "ln",
        "patience": 50,
        "periodicity": 12,
        "norm_const": 0.4,
        "detrend": False,
        "mse_weight": 0.0,
    })

    script_dir: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    dataset: str = "cyber_trend"
    use_gpu: bool = True
    num_workers: int = 0
    seed: int = 123

    # Data splits - same as VisionTS for consistency
    train_split: float = 0.43
    val_split: float = 0.30
    test_split: float = 0.27

    # Feature selection
    use_feature_selection: bool = True
    n_features: int = 25

    # Model architecture
    model_arch: str = "mae_base"
    mode: str = "train"
    finetune_type: str = "ln"
    load_pretrained: bool = True

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 50
    temporal_weight: float = 0.3  # Weight for temporal difference loss component
    corr_weight: float = 1.0  # Weight for correlation loss component
    mse_weight: float = 0.0  # Weight for squared-error loss component (directly targets RSE)

    # Hps
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    warmup_epochs: int = 0
    augment_sigma: float = 0.0

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

    # Custom ViT
    use_custom_vit: bool = False
    customvit_layers: int = 4
    customvit_heads: int = 4
    customvit_dim: int = 128
    customvit_patch_size: int = 4
    customvit_dropout: float = 0.1
    customvit_pretrained_path: Optional[str] = None
    use_aux_direction: bool = False
    direction_loss_weight: float = 0.1

    # Detrend: per-window per-feature linear trend subtraction inside the dataset, model trains on residuals, eval/test reconstruct in original space.
    detrend: bool = False

    def __post_init__(self) -> None:
        self.script_dir = Path(__file__).resolve().parents[1]
        self._resolve_paths()
        self._apply_dataset_defaults()

    def _resolve_paths(self) -> None:
        """Resolve all path fields from self.dataset. Idempotent."""
        if self.dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. Choices: {sorted(DATASETS)}"
            )
        info = DATASETS[self.dataset]
        subdir = info["subdir"]
        self.raw_data_path = self.script_dir / info["raw_relpath"]
        self.output_dir = self.script_dir / "Processed_Data" / "visionpp" / subdir
        self.checkpoint_dir = self.output_dir / "checkpoints" / "visiontspp"
        self.ckpt_dir = self.output_dir / "ckpt"
        self.results_dir = self.script_dir / "Transformer_Pipeline" / "Results" / subdir

    def _apply_dataset_defaults(self, skip_keys=None) -> None:
        skip_keys = set(skip_keys or ())
        info = DATASETS.get(self.dataset, {})
        for key, dataset_default in info.get("defaults", {}).items():
            if key in skip_keys:
                continue
            module_default = self._MODULE_DEFAULTS.get(key)
            if module_default is None:
                continue
            if getattr(self, key) == module_default:
                setattr(self, key, dataset_default)

    def update_from_dict(self, overrides: dict) -> None:
        """Update config attributes from a dictionary of overrides.
        """
        dataset_changed = False
        provided_keys: set = set()
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                if key == "dataset" and value != self.dataset:
                    dataset_changed = True
                setattr(self, key, value)
                provided_keys.add(key)
        if dataset_changed:
            self._resolve_paths()
            self._apply_dataset_defaults(skip_keys=provided_keys)
