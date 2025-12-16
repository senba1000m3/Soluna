"""
測試聲優快取功能
此腳本測試 AniListClient 的 get_anime_voice_actors 方法的快取功能
"""

import asyncio
import sys
from datetime import datetime

from sqlmodel import Session, select

from anilist_client import AniListClient
from database import engine, init_db
from models import AnimeVoiceActorCache


async def test_voice_actor_cache():
    """測試聲優資料快取功能"""

    # 初始化資料庫
    print("🔧 初始化資料庫...")
    init_db()

    # 測試用的動漫 ID (進擊的巨人)
    test_anime_id = 16498

    print("\n" + "=" * 80)
    print("📋 測試聲優快取功能")
    print("=" * 80)

    with Session(engine) as session:
        # 清除舊的測試快取
        print(f"\n🧹 清除測試動漫 {test_anime_id} 的舊快取...")
        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.anime_id == test_anime_id
        )
        old_cache = session.exec(statement).first()
        if old_cache:
            session.delete(old_cache)
            session.commit()
            print("✅ 已清除舊快取")
        else:
            print("ℹ️  無舊快取")

        # 建立帶快取功能的 AniListClient
        client = AniListClient(db_session=session)

        # 第一次抓取 (應該從 API 抓取並儲存快取)
        print(f"\n{'='*80}")
        print("🧪 測試 1: 第一次抓取 (應該從 API 抓取)")
        print("=" * 80)
        start_time = datetime.now()

        result1 = await client.get_anime_voice_actors(test_anime_id)

        end_time = datetime.now()
        duration1 = (end_time - start_time).total_seconds()

        print(f"\n⏱️  第一次抓取耗時: {duration1:.2f} 秒")

        if result1 and "characters" in result1:
            characters = result1["characters"]["edges"]
            print(f"✅ 成功取得資料，共 {len(characters)} 個角色")

            # 顯示前 3 個角色的聲優
            print("\n📋 前 3 個角色:")
            for i, edge in enumerate(characters[:3]):
                char_name = edge["node"]["name"]["full"]
                vas = edge.get("voiceActors", [])
                if vas:
                    va_name = vas[0]["name"]["full"]
                    print(f"  {i+1}. {char_name} - CV: {va_name}")
                else:
                    print(f"  {i+1}. {char_name} - 無配音員資料")
        else:
            print("❌ 第一次抓取失敗")
            return

        # 檢查快取是否已儲存
        print(f"\n{'='*80}")
        print("🔍 檢查快取是否已儲存")
        print("=" * 80)

        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.anime_id == test_anime_id
        )
        cache_entry = session.exec(statement).first()

        if cache_entry:
            print(f"✅ 快取已儲存")
            print(f"   - 動漫 ID: {cache_entry.anime_id}")
            print(f"   - 快取時間: {cache_entry.cached_at}")
            print(f"   - 資料大小: {len(cache_entry.voice_actors_data)} 字元")
        else:
            print("❌ 快取未儲存")
            return

        # 第二次抓取 (應該從快取讀取)
        print(f"\n{'='*80}")
        print("🧪 測試 2: 第二次抓取 (應該從快取讀取)")
        print("=" * 80)

        start_time = datetime.now()

        result2 = await client.get_anime_voice_actors(test_anime_id)

        end_time = datetime.now()
        duration2 = (end_time - start_time).total_seconds()

        print(f"\n⏱️  第二次抓取耗時: {duration2:.2f} 秒")

        if result2 and "characters" in result2:
            characters = result2["characters"]["edges"]
            print(f"✅ 成功從快取讀取資料，共 {len(characters)} 個角色")
        else:
            print("❌ 第二次抓取失敗")
            return

        # 比較兩次結果
        print(f"\n{'='*80}")
        print("📊 效能比較")
        print("=" * 80)
        print(f"第一次抓取 (從 API):  {duration1:.2f} 秒")
        print(f"第二次抓取 (從快取): {duration2:.2f} 秒")

        if duration2 < duration1:
            speedup = duration1 / duration2
            time_saved = duration1 - duration2
            print(f"\n🚀 快取加速: {speedup:.2f}x 倍")
            print(f"💾 節省時間: {time_saved:.2f} 秒")
        else:
            print(f"\n⚠️  警告: 快取可能未生效")

        # 驗證資料一致性
        if result1 == result2:
            print("\n✅ 資料一致性檢查通過")
        else:
            print("\n⚠️  警告: 兩次抓取的資料不一致")

    print(f"\n{'='*80}")
    print("✅ 測試完成")
    print("=" * 80)


async def test_multiple_anime_cache():
    """測試多個動漫的快取效能"""

    print("\n\n" + "=" * 80)
    print("📋 測試多個動漫的快取效能")
    print("=" * 80)

    # 測試用的動漫 ID 列表
    test_anime_ids = [
        16498,  # 進擊的巨人
        11757,  # 刀劍神域
        20583,  # 東京喰種
        1535,   # 死亡筆記本
        5114,   # 鋼之鍊金術師 FA
    ]

    with Session(engine) as session:
        client = AniListClient(db_session=session)

        # 第一輪: 從 API 抓取
        print(f"\n🔄 第一輪: 從 API 抓取 {len(test_anime_ids)} 部動漫...")
        start_time = datetime.now()

        for i, anime_id in enumerate(test_anime_ids, 1):
            print(f"\n[{i}/{len(test_anime_ids)}] 抓取動漫 ID: {anime_id}")
            result = await client.get_anime_voice_actors(anime_id)
            if result:
                print(f"  ✅ 成功")
            else:
                print(f"  ❌ 失敗")

            # 避免觸發 API 速率限制
            await asyncio.sleep(0.3)

        end_time = datetime.now()
        duration_first = (end_time - start_time).total_seconds()

        print(f"\n⏱️  第一輪總耗時: {duration_first:.2f} 秒")
        print(f"📊 平均每部: {duration_first / len(test_anime_ids):.2f} 秒")

        # 第二輪: 從快取讀取
        print(f"\n{'='*80}")
        print(f"💾 第二輪: 從快取讀取 {len(test_anime_ids)} 部動漫...")
        start_time = datetime.now()

        for i, anime_id in enumerate(test_anime_ids, 1):
            print(f"\n[{i}/{len(test_anime_ids)}] 讀取動漫 ID: {anime_id}")
            result = await client.get_anime_voice_actors(anime_id)
            if result:
                print(f"  ✅ 成功")
            else:
                print(f"  ❌ 失敗")

        end_time = datetime.now()
        duration_second = (end_time - start_time).total_seconds()

        print(f"\n⏱️  第二輪總耗時: {duration_second:.2f} 秒")
        print(f"📊 平均每部: {duration_second / len(test_anime_ids):.2f} 秒")

        # 效能比較
        print(f"\n{'='*80}")
        print("📊 效能比較")
        print("=" * 80)
        print(f"第一輪 (從 API):   {duration_first:.2f} 秒")
        print(f"第二輪 (從快取):   {duration_second:.2f} 秒")

        if duration_second < duration_first:
            speedup = duration_first / duration_second
            time_saved = duration_first - duration_second
            print(f"\n🚀 快取加速: {speedup:.2f}x 倍")
            print(f"💾 總共節省: {time_saved:.2f} 秒")
            print(f"📉 效率提升: {((1 - duration_second/duration_first) * 100):.1f}%")

        # 檢查快取狀態
        print(f"\n{'='*80}")
        print("🔍 檢查快取狀態")
        print("=" * 80)

        for anime_id in test_anime_ids:
            statement = select(AnimeVoiceActorCache).where(
                AnimeVoiceActorCache.anime_id == anime_id
            )
            cache = session.exec(statement).first()

            if cache:
                cache_age = datetime.utcnow() - cache.cached_at
                print(f"✅ 動漫 {anime_id}: 已快取 (快取時間: {cache.cached_at}, 年齡: {cache_age.seconds} 秒)")
            else:
                print(f"❌ 動漫 {anime_id}: 無快取")


async def main():
    """主測試函數"""
    try:
        # 測試單個動漫的快取
        await test_voice_actor_cache()

        # 測試多個動漫的快取效能
        await test_multiple_anime_cache()

        print("\n\n✅ 所有測試完成！")

    except KeyboardInterrupt:
        print("\n\n⚠️  測試被使用者中斷")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 測試過程發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
