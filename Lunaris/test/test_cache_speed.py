"""
簡單的快取效能測試
直接比較使用快取前後的速度差異
"""

import asyncio
import time
from sqlmodel import Session
from database import engine, init_db
from anilist_client import AniListClient


async def test_cache_speed():
    """測試快取效能"""

    print("=" * 80)
    print("快取效能測試")
    print("=" * 80)

    # 初始化資料庫
    init_db()

    # 測試用的動漫 ID（已經有快取的）
    test_anime_ids = [103572, 99807, 21711, 21366, 98478]

    print(f"\n測試 {len(test_anime_ids)} 部動漫的抓取速度\n")

    with Session(engine) as session:
        # 建立帶快取的 client
        client_with_cache = AniListClient(db_session=session)

        # 測試 5 次，看平均速度
        print("開始測試（會執行 5 次）...\n")

        times = []

        for round_num in range(1, 6):
            print(f"第 {round_num} 次測試:")
            start_time = time.time()

            for anime_id in test_anime_ids:
                result = await client_with_cache.get_anime_voice_actors(anime_id)

            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)

            print(f"  耗時: {duration:.3f} 秒")
            print()

        # 計算平均
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("=" * 80)
        print("測試結果:")
        print("=" * 80)
        print(f"平均耗時: {avg_time:.3f} 秒")
        print(f"最快:     {min_time:.3f} 秒")
        print(f"最慢:     {max_time:.3f} 秒")
        print(f"\n每部動漫平均: {avg_time / len(test_anime_ids):.3f} 秒")

        print("\n" + "=" * 80)
        print("說明:")
        print("=" * 80)
        print("如果耗時都在 0.01-0.05 秒之間，表示快取正常運作")
        print("如果耗時都在 0.5-1.0 秒之間，表示可能在使用 API")
        print("\n請檢查上方的輸出訊息:")
        print("  - '💾 使用快取資料' = 快取運作中 ✅")
        print("  - '🎤 從 API 抓取' = 沒用到快取 ❌")


if __name__ == "__main__":
    asyncio.run(test_cache_speed())
