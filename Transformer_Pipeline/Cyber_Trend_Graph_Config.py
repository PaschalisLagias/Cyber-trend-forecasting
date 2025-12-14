from dataclasses import dataclass, field, asdict
from pathlib import Path
import torch

@dataclass
class PDFormerConfig:
    """
    Configuration dataclass for the PDFormer Graph Pipeline.
    Aligned with B-MTGNN experimental settings.
    """

    # --- Paths (Auto-Resolved in __post_init__) ---
    script_dir: Path = field(init=False)
    project_root: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    
    # --- Experimental Settings ---
    input_window: int = 10       # 10 Months History
    output_window: int = 36      # 36 Months Forecast
    train_split: float = 0.43
    val_split: float = 0.30
    # Test split is implicitly remainder (0.27)

    # --- Data Loading ---
    batch_size: int = 16         # Small batch for small dataset (163 samples)
    num_workers: int = 0         # 0 is safer for debugging/compatibility
    
    # --- Model Architecture ---
    model_name: str = "PDFormer"
    embed_dim: int = 64
    skip_dim: int = 256
    lape_dim: int = 8        # Laplacian Positional Encoding dimension
    geo_num_heads: int = 4   # Geometric Attention Heads
    sem_num_heads: int = 2   # Semantic Attention Heads
    t_num_heads: int = 2     # Temporal Attention Heads
    mlp_ratio: int = 4
    qkv_bias: bool = True
    
    enc_depth: int = 2       # Reduced from 6 to prevent overfitting
    type_ln: str = "post"
    type_short_path: str = "hop"

    # --- Regularisation ---
    dropout: float = 0.2     # Increased for small dataset
    attn_drop: float = 0.2
    drop_path: float = 0.2

    # --- Attention Spans ---
    far_mask_delta: int = 2  # Reduced for shorter input window
    dtw_delta: int = 2
    s_attn_size: int = 3
    t_attn_size: int = 1

    # --- Optimisation ---
    learner: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_epoch: int = 100
    use_curriculum_learning: bool = True
    step_size: int = 100
    task_level: int = 0

    # --- Hardware ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        """
        Dynamically sets paths relative to this script's location.
        Assumes this script is in `Transformer_Pipeline/`.
        """
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        
        # Define standard paths
        self.raw_data_path = self.project_root / "Data_Preparation" / "Cyber_Trend_Forecasting_All.csv"
        self.processed_data_dir = self.project_root / "Processed_Data" / "Graph"

    def to_dict(self) -> dict:
        """Converts the config to a standard dictionary for the model wrapper."""
        return asdict(self)

    def update_from_args(self, args) -> None:
        """Updates attributes from an argparse Namespace."""
        if hasattr(args, 'batch_size') and args.batch_size:
            self.batch_size = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            self.max_epoch = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.learning_rate = args.learning_rate