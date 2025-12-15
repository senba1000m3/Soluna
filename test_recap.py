# -*- coding: utf-8 -*-
"""
Test script for Recap endpoint
"""

import asyncio
import json
import sys

import httpx

BACKEND_URL = "http://localhost:8000"


async def test_recap(username: str, year: int = None):
    """
    測試 Recap endpoint

    Args:
        username: AniList 使用者名稱
        year: 年份 (None 表示全部時間)
    """
    print("=" * 70)
    print(f"🧪 測試 Recap 功能")
    print(f"   使用者: {username}")
    print(f"   年份: {year if year else '全部時間'}")
    print("=" * 70)

    payload = {"username": username}
    if year:
        payload["year"] = year

    print(f"\n📤 發送請求到: {BACKEND_URL}/recap")
    print(f"📝 Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("\n⏳ 等待回應...")
            response = await client.post(
                f"{BACKEND_URL}/recap",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            print(f"\n📥 收到回應")
            print(f"   狀態碼: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type')}")

            if response.status_code != 200:
                print(f"\n❌ 請求失敗!")
                print(f"   錯誤訊息: {response.text}")
                return False

            data = response.json()

            print(f"\n✅ 請求成功!")
            print(f"\n📊 Recap 統計:")
            print(f"   使用者: {data.get('username')}")
            print(f"   年份: {data.get('year', '全部')}")
            print(f"   是否全部時間: {data.get('is_all_time')}")
            print(f"   總動漫數: {data.get('total_anime')}")
            print(f"   總集數: {data.get('total_episodes')}")
            print(f"   總時長: {data.get('total_hours')} 小時")
            print(f"   完成數: {data.get('completed_count')}")
            print(f"   觀看中: {data.get('watching_count')}")
            print(f"   棄番數: {data.get('dropped_count')}")
            print(f"   計劃中: {data.get('planned_count')}")
            print(f"   平均評分: {data.get('average_score')}")
            print(f"   評分總數: {data.get('total_scored')}")
            print(f"   成就數: {len(data.get('achievements', []))}")

            if data.get("achievements"):
                print(f"\n🏆 成就列表:")
                for achievement in data.get("achievements", []):
                    print(
                        f"   {achievement['icon']} {achievement['title']}: {achievement['description']}"
                    )

            if data.get("top_anime"):
                print(f"\n⭐ Top 5 動漫:")
                for i, anime in enumerate(data.get("top_anime", [])[:5], 1):
                    title = anime.get("title_english") or anime.get("title")
                    print(f"   {i}. {title} (評分: {anime.get('score')})")

            if data.get("genre_distribution"):
                print(f"\n🎭 Top 5 類型:")
                genre_items = list(data.get("genre_distribution", {}).items())[:5]
                for genre, count in genre_items:
                    print(f"   {genre}: {count} 部")

            print(f"\n✅ 測試完成!")
            print("=" * 70)
            return True

    except httpx.TimeoutException:
        print(f"\n❌ 請求超時 (>60秒)")
        print("   請檢查:")
        print("   1. 後端是否正在運行")
        print("   2. AniList API 是否回應正常")
        print("   3. 使用者是否有大量動漫資料")
        return False
    except httpx.ConnectError:
        print(f"\n❌ 無法連接到後端")
        print(f"   請確認後端正在運行於 {BACKEND_URL}")
        print(f"   啟動指令: cd Lunaris && uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"\n❌ 發生錯誤: {str(e)}")
        print(f"   錯誤類型: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


async def test_health():
    """測試後端健康狀態"""
    print("\n🏥 檢查後端健康狀態...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                print("✅ 後端運行正常")
                return True
            else:
                print(f"⚠️  後端回應異常 (狀態碼: {response.status_code})")
                return False
    except Exception as e:
        print(f"❌ 無法連接到後端: {str(e)}")
        return False


async def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("Soluna Recap Test Tool")
    print("=" * 70)

    # Check backend
    if not await test_health():
        print("\nPlease start backend first:")
        print("   cd Lunaris")
        print("   uvicorn main:app --reload")
        return

    # Test cases
    test_cases = [
        ("senba1000m3", None, "Test all-time Recap"),
        ("senba1000m3", 2024, "Test 2024 Recap"),
        ("senba1000m3", 2023, "Test 2023 Recap"),
    ]

    # Use custom test if command line args provided
    if len(sys.argv) > 1:
        username = sys.argv[1]
        year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        test_cases = [(username, year, f"Custom test: {username}")]

    results = []
    for username, year, description in test_cases:
        print(f"\nTest: {description}")
        success = await test_recap(username, year)
        results.append((description, success))
        print("\n" + "-" * 70)
        await asyncio.sleep(1)  # Avoid API rate limit

    # Show test results summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    for desc, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} - {desc}")

    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    print("\nUsage:")
    print("  python test_recap.py                    # Run default tests")
    print("  python test_recap.py <username>         # Test specific user (all-time)")
    print("  python test_recap.py <username> <year>  # Test specific user and year")
    print()

    asyncio.run(main())
