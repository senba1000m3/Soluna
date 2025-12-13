"""
BERT Recommender Configuration
配置 BERT 模型的路徑和參數
"""

import os
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent

# BERT 模型相關路徑
BERT_MODEL_DIR = PROJECT_ROOT / "data" / "bert_model"
BERT_MODEL_PATH = BERT_MODEL_DIR / "pretrained_bert.pth"
BERT_DATASET_PATH = BERT_MODEL_DIR / "dataset.pkl"
BERT_METADATA_PATH = BERT_MODEL_DIR / "animes.json"

# ID 映射檔案（AniList ID <-> Dataset ID）
ID_MAPPING_PATH = BERT_MODEL_DIR / "id_mapping.json"

# BERT 模型參數
BERT_CONFIG = {
    "max_seq_length": 200,  # 最大序列長度
    "hidden_size": 256,  # 隱藏層大小
    "num_attention_heads": 4,  # 注意力頭數量
    "num_hidden_layers": 2,  # Transformer 層數
    "dropout": 0.1,  # Dropout 率
}

# 推薦引擎參數
RECOMMENDATION_CONFIG = {
    # 是否啟用 BERT 推薦
    "use_bert": True,
    # BERT 推薦權重 (0.0 - 1.0)
    # 如果 use_bert=False，會自動使用 100% content-based
    "bert_weight": 0.6,
    "content_weight": 0.4,
    # 從 BERT 取前 K 個推薦作為參考
    "top_reference_anime": 50,
    # 運算設備 ('cpu' 或 'cuda')
    "device": "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu",
}

# 資料下載 URLs (從 AnimeRecBERT 專案)
DOWNLOAD_URLS = {
    "model": "https://www.kaggle.com/api/v1/datasets/download/ramazanturann/animeratings-mini-54m",
    "dataset": "https://www.kaggle.com/api/v1/datasets/download/ramazanturann/animeratings-mini-54m",
}


def ensure_bert_directories():
    """
    確保所有必要的目錄都存在
    """
    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return BERT_MODEL_DIR


def check_bert_files_exist() -> dict:
    """
    檢查 BERT 模型檔案是否存在

    Returns:
        字典，包含各檔案的存在狀態
    """
    return {
        "model": BERT_MODEL_PATH.exists(),
        "dataset": BERT_DATASET_PATH.exists(),
        "metadata": BERT_METADATA_PATH.exists(),
        "id_mapping": ID_MAPPING_PATH.exists(),
    }


def get_bert_availability() -> tuple[bool, str]:
    """
    檢查 BERT 模型是否可用

    Returns:
        (是否可用, 狀態訊息)
    """
    files = check_bert_files_exist()

    if all(files.values()):
        return True, "All BERT model files are available"

    missing = [name for name, exists in files.items() if not exists]
    return False, f"Missing BERT files: {', '.join(missing)}"


def get_recommendation_mode() -> str:
    """
    獲取當前推薦模式

    Returns:
        'hybrid' (BERT + Content) 或 'content_only'
    """
    if not RECOMMENDATION_CONFIG["use_bert"]:
        return "content_only"

    available, _ = get_bert_availability()
    if not available:
        return "content_only"

    return "hybrid"


if __name__ == "__main__":
    # 測試配置
    print("BERT Configuration Check")
    print("=" * 50)

    ensure_bert_directories()
    print(f"BERT Model Directory: {BERT_MODEL_DIR}")

    files = check_bert_files_exist()
    print("\nFile Status:")
    for name, exists in files.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {exists}")

    available, message = get_bert_availability()
    print(f"\nBERT Availability: {message}")

    mode = get_recommendation_mode()
    print(f"Recommendation Mode: {mode}")

    if mode == "hybrid":
        print(
            f"\nWeights: BERT {RECOMMENDATION_CONFIG['bert_weight'] * 100}% + Content {RECOMMENDATION_CONFIG['content_weight'] * 100}%"
        )
