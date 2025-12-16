"""
測試 XGBoost 訓練是否正確使用數據
模擬 DropPredict 的訓練流程
"""

import asyncio

from sqlmodel import Session, create_engine, select

from anilist_client import AniListClient
from database import init_db
from drop_analysis_engine import DropAnalysisEngine
from ingest_data import fetch_and_store_user_data
from models import Anime, User, UserRating

DB_URL = "sqlite:///anime.db"


async def test_xgboost_training(username: str = "senba1000m3"):
    """
    測試 XGBoost 訓練流程

    這個測試會:
    1. 抓取使用者資料
    2. 訓練 XGBoost 模型
    3. 檢查是否正確使用了數據
    4. 進行預測測試
    """
    print("\n" + "=" * 80)
    print("[TEST] XGBoost Training Flow Test")
    print("=" * 80)

    # 初始化資料庫
    print("\n[INIT] Initializing database...")
    init_db()
    print("[OK] Database tables created/verified\n")

    engine = create_engine(DB_URL, echo=False)
    anilist_client = AniListClient()

    with Session(engine) as session:
        print(f"\n[STEP 1] Fetching user data ({username})")
        print("-" * 80)

        # 檢查使用者是否存在
        profile = await anilist_client.get_user_profile(username)
        if not profile:
            print(f"[ERROR] User {username} not found")
            return

        print(f"[OK] User ID: {profile.get('id')}")
        print(f"[OK] Username: {profile.get('name')}")

        # 抓取並儲存資料
        await fetch_and_store_user_data(session, username)

        # 檢查儲存的資料
        db_user = session.exec(select(User).where(User.username == username)).first()

        if not db_user:
            print("[ERROR] User data not saved correctly")
            return

        print(f"[OK] Database user ID: {db_user.id}")

        # 統計資料
        all_ratings = session.exec(
            select(UserRating).where(UserRating.user_id == db_user.id)
        ).all()

        dropped_count = sum(1 for r in all_ratings if r.status == "DROPPED")
        completed_count = sum(1 for r in all_ratings if r.status == "COMPLETED")
        current_count = sum(1 for r in all_ratings if r.status == "CURRENT")
        planning_count = sum(1 for r in all_ratings if r.status == "PLANNING")

        print(f"\n[STATS] Data Statistics:")
        print(f"  Total records: {len(all_ratings)}")
        print(f"  DROPPED: {dropped_count}")
        print(f"  COMPLETED: {completed_count}")
        print(f"  CURRENT: {current_count}")
        print(f"  PLANNING: {planning_count}")

        if dropped_count + completed_count < 10:
            print("\n[WARN] Too few DROPPED + COMPLETED records, may not be able to train")

        # 步驟 2: 訓練模型
        print(f"\n[STEP 2] Training XGBoost Model")
        print("-" * 80)

        drop_engine = DropAnalysisEngine()
        train_result = drop_engine.train_model(session, user_id=db_user.id)

        print(f"\n[RESULT] Training Results:")
        print(f"  Accuracy: {train_result.get('accuracy', 0):.2%}")
        print(f"  Sample size: {train_result.get('sample_size', 0)}")
        print(f"  Dropped: {train_result.get('dropped_count', 0)}")
        print(f"  Completed: {train_result.get('completed_count', 0)}")

        # 檢查是否真的訓練了
        if not drop_engine.is_trained:
            print("\n[ERROR] Model not trained successfully")
            return

        print("\n[OK] Model trained successfully")

        # 顯示重要特徵
        if train_result.get("top_features"):
            print(f"\n[FEATURES] Top 10 Important Features:")
            for i, (feat, imp) in enumerate(train_result["top_features"][:10], 1):
                print(f"  {i:2d}. {feat:40s}: {imp:.6f}")

        # 步驟 3: 測試預測
        print(f"\n🔮 步驟 3: 測試預測功能")
        print("-" * 80)

        # 獲取 CURRENT 動畫進行預測
        current_ratings = session.exec(
            select(UserRating)
            .where(UserRating.user_id == db_user.id)
            .where(UserRating.status == "CURRENT")
        ).all()

        if current_ratings:
            print(f"\n測試 {min(5, len(current_ratings))} 部正在觀看的動畫:")
            print()

            for i, rating in enumerate(current_ratings[:5], 1):
                anime = session.get(Anime, rating.anime_id)
                if not anime:
                    continue

                drop_prob, reasons = drop_engine.predict_drop_probability(
                    anime, db_user.id, session
                )

                print(f"{i}. {anime.title_english or anime.title_romaji}")
                print(f"   棄番機率: {drop_prob:.1%}")
                print(f"   進度: {rating.progress}/{anime.episodes or '?'}")

                if reasons:
                    print(f"   風險原因:")
                    for reason in reasons[:3]:
                        print(f"     - {reason}")
                print()
        else:
            print("沒有 CURRENT 動畫可測試")

        # 步驟 4: 驗證數據使用
        print(f"\n✅ 步驟 4: 驗證數據使用")
        print("-" * 80)

        # 檢查特徵欄位數
        if drop_engine.feature_columns:
            print(f"✓ 特徵欄位數: {len(drop_engine.feature_columns)}")
            print(f"✓ 範例特徵: {drop_engine.feature_columns[:5]}")
        else:
            print("❌ 沒有特徵欄位")

        # 檢查編碼器
        if (
            hasattr(drop_engine, "mlb_genres")
            and drop_engine.mlb_genres.classes_ is not None
        ):
            print(f"✓ 類型編碼器: {len(drop_engine.mlb_genres.classes_)} 個類型")
            print(f"  範例類型: {list(drop_engine.mlb_genres.classes_[:5])}")
        else:
            print("❌ 類型編碼器未初始化")

        if (
            hasattr(drop_engine, "mlb_tags")
            and drop_engine.mlb_tags.classes_ is not None
        ):
            print(f"✓ 標籤編碼器: {len(drop_engine.mlb_tags.classes_)} 個標籤")
        else:
            print("❌ 標籤編碼器未初始化")

        if hasattr(drop_engine, "le_studio") and hasattr(
            drop_engine.le_studio, "classes_"
        ):
            print(f"✓ 製作公司編碼器: {len(drop_engine.le_studio.classes_)} 個公司")
        else:
            print("❌ 製作公司編碼器未初始化")

        # 檢查模型物件
        if drop_engine.model is not None:
            print(f"✓ XGBoost 模型: 已訓練")
            if hasattr(drop_engine.model, "n_estimators"):
                print(f"  決策樹數量: {drop_engine.model.n_estimators}")
            if hasattr(drop_engine.model, "max_depth"):
                print(f"  最大深度: {drop_engine.model.max_depth}")
        else:
            print("❌ XGBoost 模型未初始化")

        # 最終結論
        print(f"\n" + "=" * 80)
        print("📋 測試結論")
        print("=" * 80)

        all_checks = [
            (
                train_result.get("sample_size", 0) > 0,
                f"訓練樣本數: {train_result.get('sample_size', 0)}",
            ),
            (
                train_result.get("accuracy", 0) > 0,
                f"模型準確率: {train_result.get('accuracy', 0):.2%}",
            ),
            (drop_engine.is_trained, "模型已訓練"),
            (
                len(drop_engine.feature_columns) > 0,
                f"特徵數: {len(drop_engine.feature_columns)}",
            ),
            (drop_engine.model is not None, "XGBoost 模型存在"),
        ]

        passed = sum(1 for check, _ in all_checks if check)
        total = len(all_checks)

        print(f"\n通過檢查: {passed}/{total}")
        print()

        for check, desc in all_checks:
            status = "✅" if check else "❌"
            print(f"{status} {desc}")

        if passed == total:
            print("\n🎉 所有檢查通過！XGBoost 訓練正確使用了數據")
        else:
            print(f"\n⚠️  有 {total - passed} 個檢查未通過")

        print("\n" + "=" * 80)

    await anilist_client.close()


async def main():
    """主程式"""
    import sys

    username = sys.argv[1] if len(sys.argv) > 1 else "senba1000m3"

    print(f"\n使用者名稱: {username}")
    print(
        "(可以使用參數指定其他使用者，例如: uv run python test_xgboost_training.py USERNAME)"
    )

    try:
        await test_xgboost_training(username)
    except KeyboardInterrupt:
        print("\n\n中斷測試")
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
