"""
快取診斷腳本
用於檢查快取功能是否正常運作
"""

import asyncio
from sqlmodel import Session, select
from database import engine, init_db
from models import AnimeVoiceActorCache
from anilist_client import AniListClient


async def diagnose_cache():
    """診斷快取功能"""

    print("\n" + "=" * 80)
    print("🔍 快取功能診斷")
    print("=" * 80)

    # 初始化資料庫
    print("\n1️⃣ 檢查資料庫初始化...")
    try:
        init_db()
        print("   ✅ 資料庫初始化成功")
    except Exception as e:
        print(f"   ❌ 資料庫初始化失敗: {e}")
        return

    # 測試動漫 ID
    test_anime_id = 16498  # 進擊的巨人

    with Session(engine) as session:
        # 檢查是否已有快取
        print(f"\n2️⃣ 檢查動漫 {test_anime_id} 是否已有快取...")
        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.anime_id == test_anime_id
        )
        existing_cache = session.exec(statement).first()

        if existing_cache:
            print(f"   ✅ 找到快取記錄")
            print(f"      - 快取時間: {existing_cache.cached_at}")
            print(f"      - 資料大小: {len(existing_cache.voice_actors_data)} 字元")

            # 刪除舊快取以便測試
            print(f"\n   🗑️  刪除舊快取以便測試...")
            session.delete(existing_cache)
            session.commit()
            print(f"   ✅ 已刪除")
        else:
            print(f"   ℹ️  無快取記錄（這是正常的）")

        # 建立帶快取的 client
        print(f"\n3️⃣ 建立帶快取功能的 AniListClient...")
        client = AniListClient(db_session=session)

        if client.db_session is None:
            print("   ❌ db_session 為 None，快取功能未啟用！")
            return
        else:
            print("   ✅ db_session 已設定，快取功能已啟用")

        # 第一次抓取
        print(f"\n4️⃣ 第一次抓取（應該從 API 抓取並儲存快取）...")
        print("-" * 80)

        try:
            result1 = await client.get_anime_voice_actors(test_anime_id)

            if result1:
                print(f"\n   ✅ 第一次抓取成功")
                if "characters" in result1:
                    char_count = len(result1["characters"].get("edges", []))
                    print(f"      - 角色數量: {char_count}")
            else:
                print(f"   ❌ 第一次抓取失敗")
                return
        except Exception as e:
            print(f"   ❌ 第一次抓取發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return

        # 檢查快取是否已儲存
        print(f"\n5️⃣ 檢查快取是否已儲存到資料庫...")
        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.anime_id == test_anime_id
        )
        cache_entry = session.exec(statement).first()

        if cache_entry:
            print(f"   ✅ 快取已成功儲存！")
            print(f"      - 動漫 ID: {cache_entry.anime_id}")
            print(f"      - 快取時間: {cache_entry.cached_at}")
            print(f"      - 資料大小: {len(cache_entry.voice_actors_data)} 字元")
        else:
            print(f"   ❌ 快取未儲存到資料庫！")
            print(f"   ⚠️  這是問題所在 - 快取儲存邏輯可能有問題")
            return

        # 第二次抓取
        print(f"\n6️⃣ 第二次抓取（應該從快取讀取）...")
        print("-" * 80)

        try:
            result2 = await client.get_anime_voice_actors(test_anime_id)

            if result2:
                print(f"\n   ✅ 第二次抓取成功")
                if "characters" in result2:
                    char_count = len(result2["characters"].get("edges", []))
                    print(f"      - 角色數量: {char_count}")
            else:
                print(f"   ❌ 第二次抓取失敗")
                return
        except Exception as e:
            print(f"   ❌ 第二次抓取發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return

        # 驗證兩次結果是否一致
        print(f"\n7️⃣ 驗證兩次結果是否一致...")
        if result1 == result2:
            print(f"   ✅ 兩次結果一致，快取資料正確！")
        else:
            print(f"   ⚠️  兩次結果不一致")

        # 檢查所有快取
        print(f"\n8️⃣ 檢查資料庫中所有快取記錄...")
        statement = select(AnimeVoiceActorCache)
        all_caches = session.exec(statement).all()

        print(f"   總共有 {len(all_caches)} 筆快取記錄")
        for cache in all_caches[:5]:  # 只顯示前 5 筆
            print(f"   - 動漫 {cache.anime_id}: {cache.cached_at}")

    print("\n" + "=" * 80)
    print("✅ 診斷完成")
    print("=" * 80)

    # 提供建議
    print("\n💡 診斷結果:")
    print("   如果看到 '💾 [AniList Client] 使用快取資料'，表示快取正常運作")
    print("   如果看到 '🎤 [AniList Client] 從 API 抓取'，表示使用 API")
    print("   如果第二次仍從 API 抓取，請檢查:")
    print("   1. session 是否正確傳遞")
    print("   2. 快取儲存是否成功")
    print("   3. 資料庫連線是否正常")


async def check_cache_in_recap():
    """模擬 recap 端點的快取使用"""

    print("\n" + "=" * 80)
    print("🔍 模擬 Recap 端點的快取使用")
    print("=" * 80)

    test_anime_ids = [16498, 11757, 20583]  # 測試 3 部動漫

    with Session(engine) as session:
        # 這裡模擬 recap 端點的寫法
        client_with_cache = AniListClient(db_session=session)

        print(f"\n測試 {len(test_anime_ids)} 部動漫...")

        for i, anime_id in enumerate(test_anime_ids, 1):
            print(f"\n[{i}/{len(test_anime_ids)}] 動漫 ID: {anime_id}")
            print("-" * 40)

            try:
                result = await client_with_cache.get_anime_voice_actors(anime_id)
                if result:
                    print(f"✅ 成功")
                else:
                    print(f"❌ 失敗")
            except Exception as e:
                print(f"❌ 錯誤: {e}")

            await asyncio.sleep(0.3)  # 避免速率限制

        print(f"\n第一輪完成！現在檢查快取...")

        # 檢查快取
        for anime_id in test_anime_ids:
            statement = select(AnimeVoiceActorCache).where(
                AnimeVoiceActorCache.anime_id == anime_id
            )
            cache = session.exec(statement).first()

            if cache:
                print(f"✅ 動漫 {anime_id}: 已快取")
            else:
                print(f"❌ 動漫 {anime_id}: 未快取")

        print(f"\n第二輪測試（應該全部從快取讀取）...")

        for i, anime_id in enumerate(test_anime_ids, 1):
            print(f"\n[{i}/{len(test_anime_ids)}] 動漫 ID: {anime_id}")
            print("-" * 40)

            try:
                result = await client_with_cache.get_anime_voice_actors(anime_id)
                if result:
                    print(f"✅ 成功")
                else:
                    print(f"❌ 失敗")
            except Exception as e:
                print(f"❌ 錯誤: {e}")


async def main():
    """主函數"""
    try:
        # 執行基本診斷
        await diagnose_cache()

        # 執行 recap 模擬測試
        print("\n\n")
        await check_cache_in_recap()

    except KeyboardInterrupt:
        print("\n\n⚠️  測試被使用者中斷")
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
