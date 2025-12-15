# -*- coding: utf-8 -*-
"""
Test script for voice actor data fetching
"""

import asyncio
import json
import sys

import httpx

BACKEND_URL = "http://localhost:8000"


async def test_voice_actors(username: str, year: int = None):
    """
    測試 Recap endpoint 的聲優資料

    Args:
        username: AniList 使用者名稱
        year: 年份 (None 表示全部時間)
    """
    print("=" * 70)
    print(f"Voice Actor Data Test")
    print(f"   User: {username}")
    print(f"   Year: {year if year else 'All-time'}")
    print("=" * 70)

    payload = {"username": username}
    if year:
        payload["year"] = year

    print(f"\nSending request to: {BACKEND_URL}/recap")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            print("\nWaiting for response...")
            response = await client.post(
                f"{BACKEND_URL}/recap",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            print(f"\nReceived response")
            print(f"   Status code: {response.status_code}")

            if response.status_code != 200:
                print(f"\nRequest failed!")
                print(f"   Error: {response.text}")
                return False

            data = response.json()

            print(f"\nRequest successful!")

            # Check voice actor data
            va_distribution = data.get("voice_actor_distribution", {})
            print(f"\nVoice Actor Statistics:")
            print(f"   Total found: {len(va_distribution)} voice actors")

            if va_distribution:
                print(f"\nTop 20 Voice Actors:")
                for i, (name, va_info) in enumerate(
                    list(va_distribution.items())[:20], 1
                ):
                    print(f"\n   {i}. {name}")
                    print(f"      - ID: {va_info.get('id')}")
                    print(f"      - Native: {va_info.get('native')}")
                    print(f"      - Count: {va_info.get('count')}")
                    print(f"      - Image: {'YES' if va_info.get('image') else 'NO'}")
                    if va_info.get("image"):
                        print(f"        {va_info.get('image')[:80]}...")
                    print(f"      - URL: {va_info.get('siteUrl')}")

            # Check studio data
            studio_distribution = data.get("studio_distribution", {})
            print(f"\nStudio Statistics:")
            print(f"   Total found: {len(studio_distribution)} studios")

            if studio_distribution:
                print(f"\nTop 10 Studios:")
                for i, (name, studio_info) in enumerate(
                    list(studio_distribution.items())[:10], 1
                ):
                    print(f"\n   {i}. {name}")
                    print(f"      - ID: {studio_info.get('id')}")
                    print(f"      - Count: {studio_info.get('count')}")
                    print(f"      - URL: {studio_info.get('siteUrl')}")

            # Check other statistics
            print(f"\nOther Statistics:")
            print(f"   - Tags: {len(data.get('tag_distribution', {}))}")
            print(f"   - Seasons: {len(data.get('season_distribution', {}))}")
            print(f"   - Rewatched: {len(data.get('most_rewatched', []))}")
            print(f"   - Monthly Rep: {len(data.get('monthly_representative', {}))}")

            # Show Top 5 Tags
            if data.get("tag_distribution"):
                print(f"\nTop 5 Tags:")
                for i, (tag, count) in enumerate(
                    list(data.get("tag_distribution", {}).items())[:5], 1
                ):
                    print(f"   {i}. {tag}: {count} 部")

            # Show most rewatched
            if data.get("most_rewatched"):
                print(f"\nMost Rewatched:")
                for i, anime in enumerate(data.get("most_rewatched", [])[:5], 1):
                    title = anime.get("title_english") or anime.get("title")
                    print(f"   {i}. {title} - {anime.get('repeat_count')} times")

            # Show monthly representative
            if data.get("monthly_representative"):
                print(f"\nMonthly Representatives:")
                monthly = data.get("monthly_representative", {})
                month_names = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                for month_num, anime in sorted(
                    monthly.items(), key=lambda x: int(x[0])
                ):
                    month_idx = int(month_num) - 1
                    if 0 <= month_idx < 12:
                        title = anime.get("title_english") or anime.get("title")
                        print(
                            f"   {month_names[month_idx]}: {title} (Score: {anime.get('score')})"
                        )

            print(f"\nTest completed!")
            print("=" * 70)
            return True

    except httpx.TimeoutException:
        print(f"\nRequest timeout (>120s)")
        print("   Please check:")
        print("   1. Is backend running?")
        print("   2. Is AniList API responding?")
        print("   3. Does user have large amount of data?")
        return False
    except httpx.ConnectError:
        print(f"\nCannot connect to backend")
        print(f"   Please confirm backend is running at {BACKEND_URL}")
        print(f"   Start command: cd Lunaris && uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


async def test_health():
    """Test backend health"""
    print("\nChecking backend health...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                print("Backend is healthy")
                return True
            else:
                print(f"Backend response abnormal (status: {response.status_code})")
                return False
    except Exception as e:
        print(f"Cannot connect to backend: {str(e)}")
        return False


async def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("Soluna Voice Actor Test Tool")
    print("=" * 70)

    # Check backend
    if not await test_health():
        print("\nPlease start backend first:")
        print("   cd Lunaris")
        print("   uvicorn main:app --reload")
        return

    # Test cases
    if len(sys.argv) > 1:
        username = sys.argv[1]
        year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        await test_voice_actors(username, year)
    else:
        # Default test
        print("\nUsing default test user: senba1000m3")
        await test_voice_actors("senba1000m3", None)


if __name__ == "__main__":
    print("\nUsage:")
    print("  python test_voice_actors.py                    # Use default test")
    print(
        "  python test_voice_actors.py <username>         # Test specific user (all-time)"
    )
    print(
        "  python test_voice_actors.py <username> <year>  # Test specific user and year"
    )
    print()

    asyncio.run(main())
