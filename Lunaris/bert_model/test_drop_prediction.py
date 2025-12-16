"""
棄番預測診斷腳本
追蹤執行進度並找出卡住的地方
"""

import asyncio
import sys
import time
from datetime import datetime


def print_progress(stage: str, message: str = ""):
    """打印進度訊息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {stage}: {message}")
    sys.stdout.flush()


async def test_drop_prediction(username: str = "TheT"):
    """
    測試棄番預測完整流程
    追蹤每個階段的執行時間
    """
    print("\n" + "=" * 70)
    print(f"棄番預測診斷測試 - 使用者: {username}")
    print("=" * 70 + "\n")

    start_time = time.time()

    # 階段 1: 檢查使用者
    print_progress("階段 1", "檢查使用者是否存在")
    stage_start = time.time()

    try:
        from anilist_client import AniListClient

        client = AniListClient()
        profile = await client.get_user_profile(username)

        if not profile:
            print_progress("錯誤", f"找不到使用者 {username}")
            return

        user_id = profile.get("id")
        print_progress(
            "完成",
            f"使用者 ID: {user_id} (耗時: {time.time() - stage_start:.2f}秒)",
        )

    except Exception as e:
        print_progress("錯誤", f"檢查使用者失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 階段 2: 抓取動漫清單
    print_progress("階段 2", "抓取動漫清單...")
    stage_start = time.time()

    try:
        anime_list = await client.get_user_anime_list(username)
        print_progress(
            "完成",
            f"抓取到 {len(anime_list)} 筆記錄 (耗時: {time.time() - stage_start:.2f}秒)",
        )

        # 統計各狀態數量
        status_count = {}
        for entry in anime_list:
            status = entry.get("status", "UNKNOWN")
            status_count[status] = status_count.get(status, 0) + 1

        print("  狀態分布:")
        for status, count in sorted(
            status_count.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {status}: {count}")

    except Exception as e:
        print_progress("錯誤", f"抓取清單失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 階段 3: 儲存到資料庫
    print_progress("階段 3", "儲存到資料庫...")
    stage_start = time.time()

    try:
        from sqlmodel import Session

        from database import engine, init_db
        from ingest_data import fetch_and_store_user_data

        init_db()

        with Session(engine) as session:
            await fetch_and_store_user_data(session, username)

        print_progress("完成", f"資料已儲存 (耗時: {time.time() - stage_start:.2f}秒)")

    except Exception as e:
        print_progress("錯誤", f"儲存資料失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 階段 4: 訓練 XGBoost 模型
    print_progress("階段 4", "訓練 XGBoost 模型...")
    stage_start = time.time()

    try:
        from sqlmodel import Session, select

        from database import engine
        from drop_analysis_engine import DropAnalysisEngine
        from models import User

        with Session(engine) as session:
            db_user = session.exec(
                select(User).where(User.username == username)
            ).first()

            if not db_user:
                print_progress("錯誤", "使用者未儲存到資料庫")
                return

            drop_engine = DropAnalysisEngine()
            train_result = drop_engine.train_model(session, user_id=db_user.id)

        print_progress("完成", f"訓練完成 (耗時: {time.time() - stage_start:.2f}秒)")
        print(f"  準確率: {train_result.get('accuracy', 0):.2%}")
        print(f"  樣本數: {train_result.get('sample_size', 0)}")
        print(f"  棄番數: {train_result.get('dropped_count', 0)}")
        print(f"  完成數: {train_result.get('completed_count', 0)}")

    except Exception as e:
        print_progress("錯誤", f"訓練模型失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 階段 5: 初始化混合引擎
    print_progress("階段 5", "初始化混合預測引擎...")
    stage_start = time.time()

    try:
        from hybrid_drop_prediction_engine import HybridDropPredictionEngine

        hybrid_drop_engine = HybridDropPredictionEngine(
            bert_model_path="bert_model/trained_models/best_model.pth",
            bert_dataset_path="bert_model/trained_models/item_mappings.pkl",
            bert_db_path="bert_model/bert.db",
            bert_weight=0.2,
            xgboost_weight=0.8,
            use_bert=True,
        )

        info = hybrid_drop_engine.get_model_info()
        print_progress("完成", f"引擎初始化 (耗時: {time.time() - stage_start:.2f}秒)")
        print(f"  模式: {info['mode']}")
        print(f"  BERT 啟用: {info['bert_enabled']}")
        print(f"  BERT 權重: {info['bert_weight'] * 100:.0f}%")
        print(f"  XGBoost 權重: {info['xgboost_weight'] * 100:.0f}%")

    except Exception as e:
        print_progress("錯誤", f"初始化引擎失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 階段 6: 測試預測（CURRENT 動畫）
    print_progress("階段 6", "測試預測功能...")
    stage_start = time.time()

    try:
        from sqlmodel import Session, select

        from database import engine
        from models import Anime, User, UserRating

        with Session(engine) as session:
            db_user = session.exec(
                select(User).where(User.username == username)
            ).first()

            # 訓練混合引擎的 XGBoost 部分
            train_result = hybrid_drop_engine.train_xgboost_model(
                session, user_id=db_user.id
            )

            # 找幾個 CURRENT 動畫來測試
            current_ratings = session.exec(
                select(UserRating)
                .where(UserRating.user_id == db_user.id)
                .where(UserRating.status == "CURRENT")
            ).all()[:3]

            if not current_ratings:
                print("  沒有 CURRENT 動畫可測試，嘗試 PLANNING...")
                current_ratings = session.exec(
                    select(UserRating)
                    .where(UserRating.user_id == db_user.id)
                    .where(UserRating.status == "PLANNING")
                ).all()[:3]

            if current_ratings:
                print(f"\n  測試 {len(current_ratings)} 部動畫:")
                for rating in current_ratings:
                    anime = session.get(Anime, rating.anime_id)
                    if not anime:
                        continue

                    pred_start = time.time()
                    drop_prob, reasons = hybrid_drop_engine.predict_drop_probability(
                        anime, db_user.id, session
                    )
                    pred_time = time.time() - pred_start

                    print(f"\n  動畫: {anime.title_romaji or anime.title_english}")
                    print(f"    棄番機率: {drop_prob:.1%}")
                    print(f"    預測耗時: {pred_time:.2f}秒")
                    if reasons:
                        print(f"    原因: {reasons[0]}")
            else:
                print("  沒有可測試的動畫")

        print_progress(
            "完成", f"預測測試完成 (耗時: {time.time() - stage_start:.2f}秒)"
        )

    except Exception as e:
        print_progress("錯誤", f"預測測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 總結
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("診斷完成")
    print("=" * 70)
    print(f"總耗時: {total_time:.2f}秒 ({total_time / 60:.1f}分鐘)")
    print("\n建議:")
    if total_time > 300:
        print("  - 執行時間超過 5 分鐘，可能需要優化")
    if total_time > 60:
        print("  - 如果是 BERT 推薦導致緩慢，可以:")
        print("    1. 降低 BERT 權重 (例如 0.5)")
        print("    2. 暫時停用 BERT (use_bert=False)")
        print("    3. 增加快取機制")
    else:
        print("  - 執行時間正常")

    # AniListClient 不需要手動 close
    pass


async def main():
    import sys

    username = sys.argv[1] if len(sys.argv) > 1 else "TheT"

    print(f"\n目標使用者: {username}")
    print("(可使用參數指定其他使用者，例如: python test_drop_prediction.py USERNAME)")

    try:
        await test_drop_prediction(username)
    except KeyboardInterrupt:
        print("\n\n測試已中斷")
    except Exception as e:
        print(f"\n\n測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
