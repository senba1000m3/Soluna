"""
BERT4Rec 動畫推薦模型訓練套件

這個套件包含完整的 BERT4Rec 模型訓練架構，包括：
- 數據準備和載入
- 模型訓練
- Top-K 準確率計算
- 訓練過程視覺化

主要模組：
- config: 訓練配置管理
- train_model: 主訓練腳本
- visualize: 視覺化工具
- prepare_dataset: 動畫數據準備
- load_users: 用戶數據載入

快速開始：
    1. 執行 setup.bat 設置環境
    2. 執行 run_all.bat 進行完整訓練

    或分步執行：
    1. python prepare_dataset.py --count 3000
    2. python load_users.py
    3. python train_model.py --epochs 200
"""

__version__ = "1.0.0"
__author__ = "Soluna Team"
__description__ = "BERT4Rec 動畫推薦模型訓練套件"

# 導出主要類和函數
from .config import Config, DataConfig, PathConfig, TrainingConfig
from .visualize import TrainingVisualizer, plot_final_results

__all__ = [
    "Config",
    "TrainingConfig",
    "DataConfig",
    "PathConfig",
    "TrainingVisualizer",
    "plot_final_results",
]
