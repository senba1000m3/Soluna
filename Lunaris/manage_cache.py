"""
聲優快取管理工具
提供命令列介面來管理動漫聲優資料的快取
"""

import argparse
import sys
from datetime import datetime, timedelta

from sqlmodel import Session, select

from database import engine, init_db
from models import AnimeVoiceActorCache


def list_all_caches():
    """列出所有快取"""
    print("\n" + "=" * 80)
    print("📋 所有快取記錄")
    print("=" * 80)

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache)
        caches = session.exec(statement).all()

        if not caches:
            print("\n❌ 沒有快取記錄")
            return

        print(f"\n共 {len(caches)} 筆快取記錄:\n")

        # 按快取時間排序
        caches = sorted(caches, key=lambda x: x.cached_at, reverse=True)

        for i, cache in enumerate(caches, 1):
            age = datetime.utcnow() - cache.cached_at
            days = age.days
            hours = age.seconds // 3600
            minutes = (age.seconds % 3600) // 60

            size_kb = len(cache.voice_actors_data) / 1024

            print(f"{i}. 動漫 ID: {cache.anime_id}")
            print(f"   快取時間: {cache.cached_at}")
            print(f"   快取年齡: {days} 天 {hours} 小時 {minutes} 分鐘")
            print(f"   資料大小: {size_kb:.2f} KB")
            print()


def show_cache_stats():
    """顯示快取統計資訊"""
    print("\n" + "=" * 80)
    print("📊 快取統計資訊")
    print("=" * 80)

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache)
        caches = session.exec(statement).all()

        if not caches:
            print("\n❌ 沒有快取記錄")
            return

        total_count = len(caches)
        total_size = sum(len(c.voice_actors_data) for c in caches)
        total_size_mb = total_size / (1024 * 1024)

        # 計算年齡分布
        now = datetime.utcnow()
        age_distribution = {
            "< 1 天": 0,
            "1-7 天": 0,
            "7-30 天": 0,
            "> 30 天": 0,
        }

        for cache in caches:
            age_days = (now - cache.cached_at).days
            if age_days < 1:
                age_distribution["< 1 天"] += 1
            elif age_days < 7:
                age_distribution["1-7 天"] += 1
            elif age_days < 30:
                age_distribution["7-30 天"] += 1
            else:
                age_distribution["> 30 天"] += 1

        # 找出最新和最舊的快取
        newest = max(caches, key=lambda x: x.cached_at)
        oldest = min(caches, key=lambda x: x.cached_at)

        print(f"\n總快取數量: {total_count}")
        print(f"總快取大小: {total_size_mb:.2f} MB")
        print(f"平均大小: {total_size / total_count / 1024:.2f} KB")
        print(f"\n最新快取: 動漫 {newest.anime_id} ({newest.cached_at})")
        print(f"最舊快取: 動漫 {oldest.anime_id} ({oldest.cached_at})")

        print("\n快取年齡分布:")
        for age_range, count in age_distribution.items():
            percentage = (count / total_count) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {age_range:10s}: {count:4d} ({percentage:5.1f}%) {bar}")


def delete_cache_by_id(anime_id: int):
    """刪除指定動漫的快取"""
    print(f"\n🗑️  刪除動漫 {anime_id} 的快取...")

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.anime_id == anime_id
        )
        cache = session.exec(statement).first()

        if cache:
            session.delete(cache)
            session.commit()
            print(f"✅ 成功刪除動漫 {anime_id} 的快取")
        else:
            print(f"❌ 找不到動漫 {anime_id} 的快取")


def delete_expired_caches(days: int = 30):
    """刪除過期的快取"""
    print(f"\n🗑️  刪除超過 {days} 天的快取...")

    expiry_date = datetime.utcnow() - timedelta(days=days)

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.cached_at < expiry_date
        )
        expired_caches = session.exec(statement).all()

        if expired_caches:
            print(f"找到 {len(expired_caches)} 筆過期快取:")
            for cache in expired_caches:
                age = datetime.utcnow() - cache.cached_at
                print(f"  - 動漫 {cache.anime_id} (快取時間: {cache.cached_at}, 年齡: {age.days} 天)")

            confirm = input(f"\n確定要刪除這 {len(expired_caches)} 筆快取嗎? (y/N): ")
            if confirm.lower() == "y":
                for cache in expired_caches:
                    session.delete(cache)
                session.commit()
                print(f"✅ 成功刪除 {len(expired_caches)} 筆過期快取")
            else:
                print("❌ 取消刪除")
        else:
            print(f"✅ 沒有超過 {days} 天的快取")


def delete_all_caches():
    """刪除所有快取"""
    print("\n⚠️  刪除所有快取...")

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache)
        caches = session.exec(statement).all()

        if not caches:
            print("❌ 沒有快取記錄")
            return

        print(f"找到 {len(caches)} 筆快取")

        confirm = input(f"\n⚠️  確定要刪除所有 {len(caches)} 筆快取嗎? 此操作無法復原! (y/N): ")
        if confirm.lower() == "y":
            for cache in caches:
                session.delete(cache)
            session.commit()
            print(f"✅ 成功刪除所有 {len(caches)} 筆快取")
        else:
            print("❌ 取消刪除")


def show_cache_detail(anime_id: int):
    """顯示特定動漫的快取詳細資訊"""
    print(f"\n🔍 查看動漫 {anime_id} 的快取詳情...")

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache).where(
            AnimeVoiceActorCache.anime_id == anime_id
        )
        cache = session.exec(statement).first()

        if not cache:
            print(f"❌ 找不到動漫 {anime_id} 的快取")
            return

        age = datetime.utcnow() - cache.cached_at
        size_kb = len(cache.voice_actors_data) / 1024

        print("\n" + "=" * 80)
        print(f"動漫 ID: {cache.anime_id}")
        print("=" * 80)
        print(f"快取時間: {cache.cached_at}")
        print(f"快取年齡: {age.days} 天 {age.seconds // 3600} 小時")
        print(f"資料大小: {size_kb:.2f} KB")

        # 解析並顯示聲優數量
        import json

        try:
            data = json.loads(cache.voice_actors_data)
            if "characters" in data and "edges" in data["characters"]:
                characters = data["characters"]["edges"]
                print(f"角色數量: {len(characters)}")

                # 統計聲優
                va_set = set()
                for edge in characters:
                    vas = edge.get("voiceActors", [])
                    for va in vas:
                        if "name" in va and "full" in va["name"]:
                            va_set.add(va["name"]["full"])

                print(f"聲優數量: {len(va_set)}")

                # 顯示前 5 個角色
                print("\n前 5 個角色:")
                for i, edge in enumerate(characters[:5], 1):
                    char_name = edge["node"]["name"]["full"]
                    role = edge.get("role", "UNKNOWN")
                    vas = edge.get("voiceActors", [])

                    print(f"\n{i}. {char_name} ({role})")
                    if vas:
                        for va in vas[:1]:  # 只顯示第一個聲優
                            va_name = va["name"]["full"]
                            va_native = va["name"].get("native", "")
                            print(f"   CV: {va_name} ({va_native})")
                    else:
                        print("   CV: 無")

        except Exception as e:
            print(f"⚠️  無法解析快取資料: {e}")


def export_cache_list(output_file: str):
    """匯出快取列表到檔案"""
    print(f"\n📤 匯出快取列表到 {output_file}...")

    with Session(engine) as session:
        statement = select(AnimeVoiceActorCache)
        caches = session.exec(statement).all()

        if not caches:
            print("❌ 沒有快取記錄")
            return

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("動漫ID,快取時間,快取年齡(天),資料大小(KB)\n")

            for cache in sorted(caches, key=lambda x: x.cached_at, reverse=True):
                age_days = (datetime.utcnow() - cache.cached_at).days
                size_kb = len(cache.voice_actors_data) / 1024

                f.write(f"{cache.anime_id},{cache.cached_at},{age_days},{size_kb:.2f}\n")

        print(f"✅ 成功匯出 {len(caches)} 筆記錄到 {output_file}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="聲優快取管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python manage_cache.py list              # 列出所有快取
  python manage_cache.py stats             # 顯示統計資訊
  python manage_cache.py show 16498        # 查看特定動漫的快取
  python manage_cache.py delete 16498      # 刪除特定動漫的快取
  python manage_cache.py clean --days 30   # 刪除 30 天以上的快取
  python manage_cache.py clear             # 刪除所有快取
  python manage_cache.py export cache.csv  # 匯出快取列表
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用指令")

    # list 指令
    subparsers.add_parser("list", help="列出所有快取")

    # stats 指令
    subparsers.add_parser("stats", help="顯示快取統計資訊")

    # show 指令
    show_parser = subparsers.add_parser("show", help="查看特定動漫的快取詳情")
    show_parser.add_argument("anime_id", type=int, help="動漫 ID")

    # delete 指令
    delete_parser = subparsers.add_parser("delete", help="刪除特定動漫的快取")
    delete_parser.add_argument("anime_id", type=int, help="動漫 ID")

    # clean 指令
    clean_parser = subparsers.add_parser("clean", help="刪除過期快取")
    clean_parser.add_argument(
        "--days", type=int, default=30, help="快取過期天數 (預設: 30)"
    )

    # clear 指令
    subparsers.add_parser("clear", help="刪除所有快取")

    # export 指令
    export_parser = subparsers.add_parser("export", help="匯出快取列表")
    export_parser.add_argument(
        "output", type=str, default="cache_list.csv", help="輸出檔案名稱"
    )

    args = parser.parse_args()

    # 初始化資料庫
    init_db()

    # 執行對應的指令
    if args.command == "list":
        list_all_caches()
    elif args.command == "stats":
        show_cache_stats()
    elif args.command == "show":
        show_cache_detail(args.anime_id)
    elif args.command == "delete":
        delete_cache_by_id(args.anime_id)
    elif args.command == "clean":
        delete_expired_caches(args.days)
    elif args.command == "clear":
        delete_all_caches()
    elif args.command == "export":
        export_cache_list(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被使用者中斷")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 發生錯誤: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
