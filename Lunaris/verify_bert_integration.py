"""
BERT 整合驗證腳本
檢查 BERT 模型是否正確整合到棄番預測系統中
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_file_exists(filepath: str, description: str) -> bool:
    """檢查檔案是否存在"""
    path = Path(filepath)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {description}: {filepath} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} (不存在)")
        return False


def test_import():
    """測試模組導入"""
    print("\n" + "=" * 80)
    print("📦 測試模組導入")
    print("=" * 80)

    try:
        from bert_model.bert_recommender_optimized import OptimizedBERTRecommender

        print("  ✓ bert_model.bert_recommender_optimized")
    except Exception as e:
        print(f"  ✗ bert_model.bert_recommender_optimized: {e}")
        return False

    try:
        from hybrid_drop_prediction_engine import HybridDropPredictionEngine

        print("  ✓ hybrid_drop_prediction_engine")
    except Exception as e:
        print(f"  ✗ hybrid_drop_prediction_engine: {e}")
        return False

    return True


def test_bert_initialization():
    """測試 BERT 推薦器初始化"""
    print("\n" + "=" * 80)
    print("🔧 測試 BERT 推薦器初始化")
    print("=" * 80)

    try:
        from sqlmodel import Session, create_engine

        from bert_model.bert_recommender_optimized import OptimizedBERTRecommender

        # 創建測試 session
        engine = create_engine("sqlite:///soluna.db")

        with Session(engine) as session:
            bert = OptimizedBERTRecommender(
                model_path="bert_model/trained_models/best_model.pth",
                dataset_path="bert_model/trained_models/item_mappings.pkl",
                db_session=session,
                device="cpu",
            )

            print("  ✓ BERT 推薦器初始化成功")
            print(f"  ✓ 裝置: {bert.device}")
            print(f"  ✓ 批次大小: {bert.batch_size}")

            return True

    except Exception as e:
        print(f"  ✗ BERT 推薦器初始化失敗: {e}")
        logger.exception(e)
        return False


def test_hybrid_engine():
    """測試混合棄番預測引擎"""
    print("\n" + "=" * 80)
    print("🤖 測試混合棄番預測引擎")
    print("=" * 80)

    try:
        from hybrid_drop_prediction_engine import HybridDropPredictionEngine

        engine = HybridDropPredictionEngine(
            bert_model_path="bert_model/trained_models/best_model.pth",
            bert_dataset_path="bert_model/trained_models/item_mappings.pkl",
            bert_db_path="bert_model/bert.db",
            bert_weight=0.8,
            xgboost_weight=0.2,
            use_bert=True,
        )

        print(f"  ✓ 混合引擎初始化成功")
        print(f"  ✓ BERT 啟用: {engine.use_bert}")
        print(f"  ✓ BERT 權重: {engine.bert_weight * 100:.0f}%")
        print(f"  ✓ XGBoost 權重: {engine.xgboost_weight * 100:.0f}%")

        # 取得模型資訊
        info = engine.get_model_info()
        print(f"\n  模型資訊:")
        print(f"    - 模式: {info['mode']}")
        print(f"    - BERT 啟用: {info['bert_enabled']}")
        print(f"    - BERT 可用: {info['bert_available']}")

        if not info["bert_available"]:
            print("\n  ⚠️  警告: BERT 推薦器無法使用")
            print("     請檢查:")
            print("     1. bert_model/trained_models/best_model.pth 是否存在")
            print("     2. bert_model/trained_models/item_mappings.pkl 是否存在")
            print("     3. bert_model/bert.db 是否存在")
            return False

        return True

    except Exception as e:
        print(f"  ✗ 混合引擎初始化失敗: {e}")
