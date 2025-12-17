"""
BERT4Rec 模型訓練配置文件
所有訓練相關的超參數和路徑設定
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """訓練配置"""

    # 訓練參數
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_ratio: float = 0.1

    # 模型參數
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 200
    dropout: float = 0.1

    # 硬體設定
    use_gpu: bool = True
    use_fp16: bool = False
    num_workers: int = 0

    # 儲存設定
    save_every_n_epochs: int = 10
    save_best_model: bool = True

    # 早停設定
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # 學習率調整
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5

    # 其他
    random_seed: int = 42
    log_interval: int = 100


@dataclass
class DataConfig:
    """資料配置"""

    # 資料庫設定
    db_name: str = "bert.db"

    # 資料集設定
    num_anime: int = 3000  # 要抓取的熱門動畫數量
    min_user_anime: int = 20  # 使用者至少要有的動畫數量
    user_file: str = "datas_user.txt"  # 使用者列表檔案

    # 序列設定
    mask_prob: float = 0.15  # BERT 遮罩機率

    # 評估設定
    top_k_list: list = None  # Top-K 準確率計算，例如 [1, 5, 10, 20]

    def __post_init__(self):
        if self.top_k_list is None:
            self.top_k_list = [1, 5, 10, 20]


@dataclass
class PathConfig:
    """路徑配置"""

    # 基礎路徑
    base_dir: Path = Path(__file__).parent

    # 資料路徑
    data_dir: Path = None
    db_path: Path = None
    user_file_path: Path = None

    # 輸出路徑
    output_dir: Path = None
    model_dir: Path = None
    log_dir: Path = None
    plot_dir: Path = None
    checkpoint_dir: Path = None

    def __post_init__(self):
        # 設定資料路徑
        self.data_dir = self.base_dir / "data"
        self.db_path = self.data_dir / "bert.db"
        self.user_file_path = self.base_dir / "datas_user.txt"

        # 設定輸出路徑
        self.output_dir = self.base_dir / "output"
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.plot_dir = self.output_dir / "plots"
        self.checkpoint_dir = self.output_dir / "checkpoints"

        # 創建所有必要的目錄
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """總配置類"""

    def __init__(self):
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.paths = PathConfig()

    def update_from_args(self, args):
        """從命令列參數更新配置"""
        if hasattr(args, "epochs"):
            self.training.epochs = args.epochs
        if hasattr(args, "batch_size"):
            self.training.batch_size = args.batch_size
        if hasattr(args, "lr"):
            self.training.learning_rate = args.lr
        if hasattr(args, "hidden_size"):
            self.training.hidden_size = args.hidden_size
        if hasattr(args, "num_layers"):
            self.training.num_layers = args.num_layers
        if hasattr(args, "num_heads"):
            self.training.num_heads = args.num_heads
        if hasattr(args, "max_seq_len"):
            self.training.max_seq_len = args.max_seq_len
        if hasattr(args, "dropout"):
            self.training.dropout = args.dropout
        if hasattr(args, "gpu"):
            self.training.use_gpu = args.gpu
        if hasattr(args, "fp16"):
            self.training.use_fp16 = args.fp16
        if hasattr(args, "seed"):
            self.training.random_seed = args.seed

    def print_config(self):
        """列印配置資訊"""
        print("\n" + "=" * 80)
        print("📋 訓練配置")
        print("=" * 80)

        print("\n🎯 訓練參數:")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Validation Ratio: {self.training.val_ratio}")

        print("\n🏗️  模型參數:")
        print(f"  Hidden Size: {self.training.hidden_size}")
        print(f"  Num Layers: {self.training.num_layers}")
        print(f"  Num Heads: {self.training.num_heads}")
        print(f"  Max Seq Length: {self.training.max_seq_len}")
        print(f"  Dropout: {self.training.dropout}")

        print("\n💻 硬體設定:")
        print(f"  Use GPU: {self.training.use_gpu}")
        print(f"  Use FP16: {self.training.use_fp16}")

        print("\n📊 資料設定:")
        print(f"  Num Anime: {self.data.num_anime}")
        print(f"  Min User Anime: {self.data.min_user_anime}")
        print(f"  Mask Probability: {self.data.mask_prob}")
        print(f"  Top-K: {self.data.top_k_list}")

        print("\n📁 路徑設定:")
        print(f"  Database: {self.paths.db_path}")
        print(f"  Output Dir: {self.paths.output_dir}")
        print(f"  Model Dir: {self.paths.model_dir}")
        print(f"  Plot Dir: {self.paths.plot_dir}")

        print("=" * 80)

    def save_to_file(self, filepath: Path):
        """儲存配置到檔案"""
        import json

        config_dict = {
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "val_ratio": self.training.val_ratio,
                "hidden_size": self.training.hidden_size,
                "num_layers": self.training.num_layers,
                "num_heads": self.training.num_heads,
                "max_seq_len": self.training.max_seq_len,
                "dropout": self.training.dropout,
            },
            "data": {
                "num_anime": self.data.num_anime,
                "min_user_anime": self.data.min_user_anime,
                "mask_prob": self.data.mask_prob,
                "top_k_list": self.data.top_k_list,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


# 預設配置
default_config = Config()
