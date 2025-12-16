"""
BERT 整合狀態檢查腳本
檢查 BERT 模型是否可以正常載入和使用
"""

import sys
from pathlib import Path


def check_paths():
    """檢查必要的檔案和路徑"""
    print("\n" + "=" * 70)
    print("檢查 BERT 檔案和路徑")
    print("=" * 70)

    checks = []

    # 檢查模型檔案
    model_path = Path("bert_model/trained_models/best_model.pth")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ 模型檔案: {model_path} ({size_mb:.1f} MB)")
        checks.append(True)
    else:
        print(f"✗ 模型檔案不存在: {model_path}")
        checks.append(False)

    # 檢查映射檔案
    mapping_path = Path("bert_model/trained_models/item_mappings.pkl")
    if mapping_path.exists():
        size_kb = mapping_path.stat().st_size / 1024
        print(f"✓ 映射檔案: {mapping_path} ({size_kb:.1f} KB)")
        checks.append(True)
    else:
        print(f"✗ 映射檔案不存在: {mapping_path}")
        checks.append(False)

    # 檢查資料庫
    db_path = Path("bert_model/bert.db")
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"✓ 資料庫檔案: {db_path} ({size_mb:.1f} MB)")
        checks.append(True)
    else:
        print(f"✗ 資料庫檔案不存在: {db_path}")
        checks.append(False)

    # 檢查 Python 模組
    recommender_path = Path("bert_model/bert_recommender_optimized.py")
    if recommender_path.exists():
        print(f"✓ 推薦器模組: {recommender_path}")
        checks.append(True)
    else:
        print(f"✗ 推薦器模組不存在: {recommender_path}")
        checks.append(False)

    return all(checks)


def check_imports():
    """檢查模組導入"""
    print("\n" + "=" * 70)
    print("檢查模組導入")
    print("=" * 70)

    try:
        from bert_model.bert_recommender_optimized import OptimizedBERTRecommender

        print("✓ OptimizedBERTRecommender 導入成功")
        return True
    except Exception as e:
        print(f"✗ 導入失敗: {e}")
        return False


def check_model_loading():
    """檢查模型載入"""
    print("\n" + "=" * 70)
    print("檢查模型載入")
    print("=" * 70)

    try:
        from sqlmodel import Session

        from bert_model.bert_recommender_optimized import OptimizedBERTRecommender
        from database import engine

        print("正在載入 BERT 模型...")
        with Session(engine) as session:
            bert = OptimizedBERTRecommender(
                model_path="bert_model/trained_models/best_model.pth",
                dataset_path="bert_model/trained_models/item_mappings.pkl",
                db_session=session,
                device="cpu",  # 強制使用 CPU 避免 GPU 問題
            )

        print("✓ BERT 模型載入成功")
        return True

    except Exception as e:
        print(f"✗ 模型載入失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_hybrid_engine():
    """檢查混合預測引擎"""
    print("\n" + "=" * 70)
    print("檢查混合棄番預測引擎")
    print("=" * 70)

    try:
        from hybrid_drop_prediction_engine import HybridDropPredictionEngine

        print("正在初始化混合引擎...")
        engine = HybridDropPredictionEngine(
            bert_model_path="bert_model/trained_models/best_model.pth",
            bert_dataset_path="bert_model/trained_models/item_mappings.pkl",
            bert_db_path="bert_model/bert.db",
            bert_weight=0.8,
            xgboost_weight=0.2,
            use_bert=True,
        )

        info = engine.get_model_info()
        print(f"✓ 混合引擎初始化成功")
        print(f"  模式: {info['mode']}")
        print(f"  BERT 啟用: {info['bert_enabled']}")
        print(f"  BERT 權重: {info['bert_weight'] * 100:.0f}%")
        print(f"  XGBoost 權重: {info['xgboost_weight'] * 100:.0f}%")

        return info["bert_enabled"]

    except Exception as e:
        print(f"✗ 混合引擎初始化失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("BERT 整合狀態檢查")
    print("=" * 70)

    results = []

    # 1. 檢查路徑
    results.append(("檔案路徑", check_paths()))

    # 2. 檢查導入
    results.append(("模組導入", check_imports()))

    # 3. 檢查模型載入
    results.append(("模型載入", check_model_loading()))

    # 4. 檢查混合引擎
    results.append(("混合引擎", check_hybrid_engine()))

    # 總結
    print("\n" + "=" * 70)
    print("檢查總結")
    print("=" * 70)

    for name, result in results:
        status = "✓ 通過" if result else "✗ 失敗"
        print(f"{status:8s} - {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n通過: {passed}/{total}")

    if passed == total:
        print("\n✅ 所有檢查通過！BERT 整合正常")
        print("\n後端應該可以使用 BERT 模型進行棄番預測")
        print("權重配置: BERT 80% + XGBoost 20%")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 個檢查失敗")
        print("\n可能的解決方案:")
        print("1. 確認模型已訓練: cd bert_model && train.bat")
        print("2. 檢查檔案路徑是否正確")
        print("3. 檢查 Python 依賴是否安裝完整")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n中斷檢查")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n執行錯誤: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
