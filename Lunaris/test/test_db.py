"""
測試資料庫連線和資料狀態
"""

from sqlmodel import Session, select

from database import engine, init_db
from models import GlobalUser, QuickID


def test_database():
    print("=" * 60)
    print("🔍 測試 Soluna 資料庫")
    print("=" * 60)

    # 初始化資料庫
    print("\n1. 初始化資料庫...")
    init_db()
    print("   ✅ 資料庫初始化完成")

    with Session(engine) as session:
        # 檢查 GlobalUser 表
        print("\n2. 檢查 GlobalUser 表...")
        global_users = session.exec(select(GlobalUser)).all()
        print(f"   📊 找到 {len(global_users)} 個主 ID:")
        for user in global_users:
            print(f"      - {user.anilist_username} (ID: {user.anilist_id})")
            print(f"        建立時間: {user.created_at}")
            print(f"        最後登入: {user.last_login}")

        # 檢查 QuickID 表
        print("\n3. 檢查 QuickID 表...")
        quick_ids = session.exec(select(QuickID)).all()
        print(f"   📊 找到 {len(quick_ids)} 個常用 ID:")
        for qid in quick_ids:
            owner = session.get(GlobalUser, qid.owner_id)
            owner_name = owner.anilist_username if owner else "未知"
            print(f"      - {qid.anilist_username} (ID: {qid.anilist_id})")
            print(f"        所屬主 ID: {owner_name}")
            print(f"        暱稱: {qid.nickname or '無'}")
            print(f"        建立時間: {qid.created_at}")

        # 統計資訊
        print("\n4. 統計資訊:")
        print(f"   總主 ID 數量: {len(global_users)}")
        print(f"   總常用 ID 數量: {len(quick_ids)}")

        # 檢查關聯
        if global_users:
            print("\n5. 檢查主 ID 與常用 ID 的關聯:")
            for user in global_users:
                user_quick_ids = session.exec(
                    select(QuickID).where(QuickID.owner_id == user.id)
                ).all()
                print(
                    f"   - {user.anilist_username} 有 {len(user_quick_ids)} 個常用 ID"
                )
                for qid in user_quick_ids:
                    print(f"     → {qid.anilist_username}")

    print("\n" + "=" * 60)
    print("✅ 資料庫檢查完成")
    print("=" * 60)


if __name__ == "__main__":
    test_database()
