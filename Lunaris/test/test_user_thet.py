"""
測試用戶 thet 的 Recap 快取功能
"""

import asyncio
from datetime import datetime
from sqlmodel import Session
from database import engine, init_db
from anilist_client import AniListClient


async def test_user_thet():
    """測試用戶 thet 的 Recap 功能"""

    print("\n" + "=" * 80)
    print("📋 測試用戶 thet 的 Recap 快取功能")
    print("=" * 80)

    username = "thet"

    # 初始化資料庫
    init_db()

    with Session(engine) as session:
        # 建立帶快取的 client
        client_with_cache = AniListClient(db_session=session)

        print(f"\n🔄 第一次抓取用戶 {username} 的動漫列表...")
        start_time = datetime.now()

        try:
            user_list = await client_with_cache.get_user_anime_list(username)

            if not user_list:
                print(f"❌ 找不到用戶 {username} 或列表為空")
                return

            print(f"✅ 成功抓取，共 {len(user_list)} 部動漫")

            # 收集所有動漫 ID
            anime_ids = [entry.get("media", {}).get("id") for entry in user_list if entry.get("media", {}).get("id")]
            print(f"📝 需要查詢聲優的動漫數量: {len(anime_ids)}")

            # 第一次：抓取所有聲優資料
            print(f"\n🎤 第一次抓取聲優資料（會從 API 抓取並快取）...")
            print("-" * 80)

            start_va_time = datetime.now()
            cached_count = 0
            api_count = 0

            for i, anime_id in enumerate(anime_ids[:20], 1):  # 測試前 20 部
                if i % 5 == 0:
                    print(f"進度: {i}/{min(20, len(anime_ids))}")

                result = await client_with_cache.get_anime_voice_actors(anime_id)

                # 檢查是否使用快取（根據 log 判斷）
                if result:
                    # 這裡無法直接判斷，但可以從輸出看到
                    pass

                await asyncio.sleep(0.1)  # 避免速率限制

            end_va_time = datetime.now()
            first_duration = (end_va_time - start_va_time).total_seconds()

            print(f"\n⏱️  第一次抓取聲優資料耗時: {first_duration:.2f} 秒")
            print(f"📊 平均每部: {first_duration / min(20, len(anime_ids)):.2f} 秒")

            # 第二次：應該全部從快取讀取
            print(f"\n💾 第二次抓取聲優資料（應該從快取讀取，速度超快）...")
            print("-" * 80)

            start_va_time = datetime.now()

            for i, anime_id in enumerate(anime_ids[:20], 1):
                if i % 5 == 0:
                    print(f"進度: {i}/{min(20, len(anime_ids))}")

                result = await client_with_cache.get_anime_voice_actors(anime_id)

            end_va_time = datetime.now()
            second_duration = (end_va_time - start_va_time).total_seconds()

            print(f"\n⏱️  第二次抓取聲優資料耗時: {second_duration:.2f} 秒")
            print(f"📊 平均每部: {second_duration / min(20, len(anime_ids)):.2f} 秒")

            # 效能比較
            print(f"\n" + "=" * 80)
            print("📊 效能比較")
            print("=" * 80)
            print(f"第一次（含 API + 快取）: {first_duration:.2f} 秒")
            print(f"第二次（純快取）:        {second_duration:.2f} 秒")

            if second_duration < first_duration:
                speedup = first_duration / second_duration
                time_saved = first_duration - second_duration
                print(f"\n🚀 快取加速: {speedup:.2f}x 倍")
                print(f"💾 節省時間: {time_saved:.2f} 秒")
                print(f"📉 效率提升: {((1 - second_duration/first_duration) * 100):.1f}%")

                # 推算全部動漫的時間
                if len(anime_ids) > 20:
                    estimated_full_first = first_duration * (len(anime_ids) / 20)
                    estimated_full_second = second_duration * (len(anime_ids) / 20)
                    print(f"\n📈 推算全部 {len(anime_ids)} 部動漫:")
                    print(f"   第一次預估: {estimated_full_first:.2f} 秒 ({estimated_full_first/60:.1f} 分鐘)")
                    print(f"   第二次預估: {estimated_full_second:.2f} 秒 ({estimated_full_second/60:.1f} 分鐘)")
                    print(f"   節省時間: {(estimated_full_first - estimated_full_second)/60:.1f} 分鐘")

        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ 測試完成")
    print("=" * 80)

    print("\n💡 注意事項:")
    print("   - 第一次查詢會比較慢，因為需要從 API 抓取並儲存快取")
    print("   - 第二次查詢會超快，因為直接從資料庫讀取快取")
    print("   - 如果兩次都很慢，請檢查 console 是否顯示 '💾 使用快取資料'")


if __name__ == "__main__":
    asyncio.run(test_user_thet())
