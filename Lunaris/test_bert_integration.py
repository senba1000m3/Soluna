"""
測試 BERT 整合到推薦系統

這個腳本會：
1. 載入訓練好的 BERT 模型
2. 測試推薦功能
3. 驗證與 hybrid_recommendation_engine 的整合
"""

import asyncio
import sys
from pathlib import Path

from sqlmodel import Session

from bert_model.bert_recommender_optimized import OptimizedBERTRecommender
from database import engine


async def test_bert_model():
    """測試 BERT 模型載入和推薦"""
    print("\n" + "=" * 80)
    print("🧪 測試 BERT 模型整合")
    print("=" * 80)

    # 檢查模型檔案
    model_path = Path("bert_model/trained_models/best_model.pth")
    mapping_path = Path("bert_model/trained_models/item_mappings.pkl")

    print("\n📁 檢查模型檔案...")
    if not model_path.exists():
        print(f"  ❌ 模型檔案不存在: {model_path}")
        return False

    if not mapping_path.exists():
        print(f"  ❌ 映射檔案不存在: {mapping_path}")
        return False

    model_size = model_path.stat().st_size / (1024 * 1024)
    mapping_size = mapping_path.stat().st_size / 1024

    print(f"  ✅ 模型檔案: {model_size:.1f} MB")
    print(f"  ✅ 映射檔案: {mapping_size:.1f} KB")

    # 載入模型
    print("\n🔧 載入 BERT 推薦器...")
    try:
        with Session(engine) as session:
            bert = OptimizedBERTRecommender(
                model_path=str(model_path),
                dataset_path=str(mapping_path),
                db_session=session,
                device="auto",
            )
            print("  ✅ BERT 推薦器載入成功")
    except Exception as e:
        print(f"  ❌ 載入失敗: {e}")
        return False

    # 測試推薦
    print("\n🎯 測試推薦功能...")
    test_anime_ids = [
        16498,  # Shingeki no Kyojin (Attack on Titan)
        1535,  # Death Note
        101922,  # Kimetsu no Yaiba (Demon Slayer)
    ]

    print(f"  測試動畫 ID: {test_anime_ids}")

    try:
        with Session(engine) as session:
            bert.db_session = session
            recommendations = bert.get_recommendations(
                user_anime_ids=test_anime_ids,
                top_k=10,
                use_anilist_ids=True,
            )

            if recommendations:
                print(f"  ✅ 成功取得 {len(recommendations)} 個推薦")
                print("\n  前 5 個推薦:")
                for i, rec in enumerate(recommendations[:5], 1):
                    title = rec.get("title", "Unknown")
                    score = rec.get("score", 0)
                    anime_id = rec.get("anime_id", 0)
                    print(f"    {i}. {title} (ID: {anime_id}, 分數: {score:.3f})")
            else:
                print("  ⚠️  沒有推薦結果")
                return False

    except Exception as e:
        print(f"  ❌ 推薦失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 測試統計
    print("\n📊 推薦器統計:")
    bert.print_stats()

    print("\n" + "=" * 80)
    print("✅ 所有測試通過！")
    print("=" * 80)

    return True


async def test_hybrid_integration():
    """測試與 hybrid_recommendation_engine 的整合"""
    print("\n" + "=" * 80)
    print("🔗 測試 Hybrid Recommendation Engine 整合")
    print("=" * 80)

    try:
        from hybrid_recommendation_engine import HybridRecommendationEngine

        print("\n🔧 初始化 Hybrid Engine...")
        engine = HybridRecommendationEngine(use_bert=True)

        if engine.use_bert and engine.bert_recommender:
            print("  ✅ BERT 已成功整合到 Hybrid Engine")
        else:
            print("  ⚠️  BERT 未啟用")
            return False

        print("\n" + "=" * 80)
        print("✅ Hybrid Engine 整合測試通過！")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"  ❌ 整合測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """主測試流程"""
    print("\n" + "=" * 80)
    print("🚀 BERT 模型整合測試")
    print("=" * 80)

    # 測試 1: BERT 模型
    bert_ok = await test_bert_model()

    if not bert_ok:
        print("\n❌ BERT 模型測試失敗")
        sys.exit(1)

    # 測試 2: Hybrid 整合
    hybrid_ok = await test_hybrid_integration()

    if not hybrid_ok:
        print("\n❌ Hybrid Engine 整合測試失敗")
        sys.exit(1)

    # 全部通過
    print("\n" + "=" * 80)
    print("🎉 所有測試通過！")
    print("=" * 80)
    print("\n✅ BERT 模型已成功整合到推薦系統")
    print("\n📝 下一步:")
    print("  1. 在 main.py 中使用 HybridRecommendationEngine")
    print("  2. 前端可以通過 /recommend API 使用 BERT 推薦")
    print("  3. 定期重新訓練模型以改善推薦品質")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被中斷")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
