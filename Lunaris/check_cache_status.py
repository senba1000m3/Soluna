"""
快取狀態檢查腳本
快速檢查特定用戶的動漫是否已有快取
"""

import sys
from sqlmodel import Session, select
from database import engine, init_db
from models import AnimeVoiceActorCache
from anilist_client import AniListClient
import asyncio


async def check_user_cache(username):
    """檢查用戶的動漫快取狀態"""

    print("=" * 80)
    print(f"檢查用戶 {username} 的快取狀態")
    print("=" * 80)

    # 初始化資料庫
    init_db()

    # 建立不帶快取的 client（只用來抓取用戶列表）
    client = AniListClient()

    print(f"\n正在抓取用戶 {username} 的動漫列表...")

    try:
        # 使用 asyncio.wait_for 設定超時
        user_list = await asyncio.wait_for(
            client.get_user_anime_list(username),
            timeout=30.0
        )

        if not user_list:
            print(f"錯誤: 找不到用戶 {username} 或列表為空")
            return

        print(f"成功! 找到 {len(user_list)} 部動漫")

        # 收集動漫 ID
        anime_ids = []
        for entry in user_list:
            media = entry.get("media", {})
            anime_id = media.get("id")
            if anime_id:
                anime_ids.append(anime_id)

        print(f"\n總共有 {len(anime_ids)} 部動漫需要檢查快取")

        # 檢查快取狀態
        with Session(engine) as session:
            cached_count = 0
            not_cached_count = 0

            print("\n檢查快取中...")

            for anime_id in anime_ids:
                statement = select(AnimeVoiceActorCache).where(
                    AnimeVoiceActorCache.anime_id == anime_id
                )
                cache = session.exec(statement).first()

                if cache:
                    cached_count += 1
                else:
                    not_cached_count += 1

            print(f"\n快取統計:")
            print(f"  已快取: {cached_count} 部 ({cached_count/len(anime_ids)*100:.1f}%)")
            print(f"  未快取: {not_cached_count} 部 ({not_cached_count/len(anime_ids)*100:.1f}%)")

            if cached_count == len(anime_ids):
                print(f"\n✅ 太棒了! 所有動漫都已快取，第二次查詢會超快!")
            elif cached_count > 0:
                print(f"\n⚡ 部分動漫已快取，第二次查詢會加快 {cached_count/len(anime_ids)*100:.0f}%")
            else:
                print(f"\n📝 所有動漫都未快取，這是第一次查詢，會需要一些時間")

            # 顯示前 10 個未快取的動漫
            if not_cached_count > 0 and not_cached_count <= 10:
                print(f"\n未快取的動漫 ID:")
                for anime_id in anime_ids:
                    statement = select(AnimeVoiceActorCache).where(
                        AnimeVoiceActorCache.anime_id == anime_id
                    )
                    cache = session.exec(statement).first()
                    if not cache:
                        print(f"  - {anime_id}")

    except asyncio.TimeoutError:
        print("錯誤: 請求超時（可能是 API 速率限制）")
        print("建議: 等待 1-2 分鐘後再試")
    except Exception as e:
        print(f"錯誤: {e}")

        # 檢查是否是速率限制
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            print("\n⚠️  API 速率限制!")
            print("   AniList API 有速率限制（每分鐘約 90 次請求）")
            print("   建議: 等待 1-2 分鐘後再試")
        else:
            import traceback
            traceback.print_exc()


async def main():
    """主函數"""

    # 檢查命令列參數
    if len(sys.argv) < 2:
        print("使用方式: python check_cache_status.py <username>")
        print("範例: python check_cache_status.py thet")
        sys.exit(1)

    username = sys.argv[1]

    try:
        await check_user_cache(username)
    except KeyboardInterrupt:
        print("\n\n操作被中斷")
    except Exception as e:
        print(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
