# coding: utf-8
"""
測試單個動漫的聲優數據結構
用於調試 AniList API 返回的 characters.edges 結構
"""

import asyncio
import json

from anilist_client import AniListClient


async def test_single_anime_voice_actors():
    """測試單個動漫的聲優數據"""
    client = AniListClient()

    # 測試一個知名動漫的 ID (進擊的巨人)
    test_anime_id = 16498

    query = """
    query ($id: Int) {
      Media(id: $id, type: ANIME) {
        id
        title {
          romaji
          english
        }
        characters(page: 1, perPage: 10, sort: ROLE) {
          edges {
            role
            node {
              id
              name {
                full
                native
              }
            }
            voiceActors(language: JAPANESE, sort: RELEVANCE) {
              id
              name {
                full
                native
              }
              image {
                large
                medium
              }
              siteUrl
            }
          }
        }
      }
    }
    """

    variables = {"id": test_anime_id}

    try:
        print("=" * 80)
        print(f"測試動漫 ID: {test_anime_id}")
        print("=" * 80)

        data = await client._post_request(query, variables)

        if data and "Media" in data:
            media = data["Media"]
            print(f"\n動漫標題: {media['title']['romaji']}")
            print(f"英文標題: {media['title'].get('english', 'N/A')}")
            print("\n" + "=" * 80)
            print("角色與聲優列表:")
            print("=" * 80)

            characters = media.get("characters", {})
            if characters and isinstance(characters, dict):
                edges = characters.get("edges", [])
                print(f"\n找到 {len(edges)} 個角色")

                if edges:
                    for i, edge in enumerate(edges[:5], 1):  # 只顯示前5個
                        print(f"\n--- 角色 {i} ---")
                        print(f"角色類型: {edge.get('role', 'UNKNOWN')}")

                        node = edge.get("node", {})
                        if node:
                            print(
                                f"角色名稱: {node.get('name', {}).get('full', 'Unknown')}"
                            )
                            print(
                                f"角色日文名: {node.get('name', {}).get('native', 'N/A')}"
                            )

                        voice_actors = edge.get("voiceActors", [])
                        if voice_actors:
                            print(f"聲優數量: {len(voice_actors)}")
                            for j, va in enumerate(voice_actors, 1):
                                va_name = va.get("name", {})
                                print(f"\n  聲優 {j}:")
                                print(f"    ID: {va.get('id')}")
                                print(f"    名稱: {va_name.get('full', 'Unknown')}")
                                print(f"    日文名: {va_name.get('native', 'N/A')}")

                                va_image = va.get("image", {})
                                if isinstance(va_image, dict):
                                    image_url = va_image.get("large") or va_image.get(
                                        "medium"
                                    )
                                    print(
                                        f"    圖片: {image_url if image_url else 'None'}"
                                    )
                                else:
                                    print(f"    圖片: None")

                                print(f"    連結: {va.get('siteUrl', 'N/A')}")
                        else:
                            print("  沒有聲優資料")
                else:
                    print("\n⚠️ edges 為空")
            else:
                print("\n❌ 沒有找到 characters 資料")

            # 輸出完整 JSON 供檢查
            print("\n" + "=" * 80)
            print("完整 characters 資料結構:")
            print("=" * 80)
            print(json.dumps(characters, indent=2, ensure_ascii=False))

        else:
            print("❌ 無法取得動漫資料")

    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback

        traceback.print_exc()


async def test_user_list_voice_actors(username: str):
    """測試使用者列表中的聲優數據"""
    client = AniListClient()

    print("=" * 80)
    print(f"測試使用者: {username}")
    print("=" * 80)

    try:
        user_list = await client.get_user_anime_list(username)
        print(f"\n成功取得 {len(user_list)} 筆動漫資料")

        # 統計聲優數據
        total_anime = len(user_list)
        anime_with_va = 0
        total_va_count = 0
        va_dict = {}

        for entry in user_list[:10]:  # 只檢查前10部
            media = entry.get("media", {})
            title = media.get("title", {}).get("romaji", "Unknown")

            characters = media.get("characters", {})
            if characters and isinstance(characters, dict):
                edges = characters.get("edges", [])
                if edges:
                    has_va = False
                    for edge in edges:
                        voice_actors = edge.get("voiceActors", [])
                        if voice_actors:
                            has_va = True
                            for va in voice_actors:
                                va_name = va.get("name", {})
                                if isinstance(va_name, dict):
                                    va_full = va_name.get("full")
                                    if va_full:
                                        total_va_count += 1
                                        va_dict[va_full] = va_dict.get(va_full, 0) + 1

                    if has_va:
                        anime_with_va += 1
                        print(f"✓ {title}: 找到聲優資料")
                    else:
                        print(f"✗ {title}: 沒有聲優資料")

        print("\n" + "=" * 80)
        print("統計結果:")
        print("=" * 80)
        print(f"檢查的動漫數: {min(10, total_anime)}")
        print(f"有聲優資料的動漫: {anime_with_va}")
        print(f"總聲優出現次數: {total_va_count}")
        print(f"不重複聲優數: {len(va_dict)}")

        if va_dict:
            print("\n前10名最常出現的聲優:")
            sorted_va = sorted(va_dict.items(), key=lambda x: x[1], reverse=True)
            for i, (name, count) in enumerate(sorted_va[:10], 1):
                print(f"{i}. {name}: {count} 次")

    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("測試選項:")
    print("1. 測試單個動漫的聲優數據結構")
    print("2. 測試使用者列表的聲優數據")

    choice = input("\n請選擇 (1/2): ").strip()

    if choice == "1":
        asyncio.run(test_single_anime_voice_actors())
    elif choice == "2":
        username = input("請輸入 AniList 使用者名稱: ").strip()
        if username:
            asyncio.run(test_user_list_voice_actors(username))
        else:
            print("❌ 使用者名稱不能為空")
    else:
        print("❌ 無效的選擇")
