"""
清理 BERT 資料庫並重新載入使用者資料

此腳本會：
1. 清除所有舊的 mock 和測試資料
2. 保留動畫資料 (BERTAnime)
3. 重新從 datas_user.txt 載入使用者資料
"""

import asyncio
import sys
from pathlib import Path

from sqlmodel import Session, create_engine, delete, select

from prepare_bert_dataset import (
    BERTAnime,
    BERTDatasetMetadata,
    BERTUserAnimeList,
    init_bert_db,
)

BERT_DB_URL = "sqlite:///bert.db"


async def clean_and_reload():
    """清理並重新載入資料"""
    print("\n" + "=" * 80)
    print("🧹 清理 BERT 資料庫")
    print("=" * 80)

    # 初始化資料庫
    init_bert_db()
    engine = create_engine(BERT_DB_URL, echo=False)

    with Session(engine) as session:
        # 檢查動畫資料
        anime_count = len(session.exec(select(BERTAnime)).all())
        print(f"\n📚 動畫資料: {anime_count} 部")

        if anime_count == 0:
            print("\n❌ 錯誤: 沒有動畫資料")
            print("   請先執行: prepare_anime.bat")
            sys.exit(1)

        # 刪除所有使用者-動畫記錄
        print("\n🗑️  刪除舊的使用者資料...")
        old_records = session.exec(select(BERTUserAnimeList)).all()
        print(f"   找到 {len(old_records)} 筆舊記錄")

        session.exec(delete(BERTUserAnimeList))
        session.commit()
        print("   ✅ 已清除所有使用者資料")

        # 清除 metadata
        session.exec(delete(BERTDatasetMetadata))
        session.commit()

    print("\n" + "=" * 80)
    print("✅ 資料庫清理完成")
    print("=" * 80)

    # 檢查 datas_user.txt
    user_file = Path("datas_user.txt")
    if not user_file.exists():
        print("\n❌ 錯誤: datas_user.txt 不存在")
        sys.exit(1)

    # 重新載入使用者資料
    print("\n" + "=" * 80)
    print("📥 重新載入使用者資料")
    print("=" * 80)
    print("\n正在執行 load_users_from_file.py...\n")

    # 動態導入並執行
    from load_users_from_file import UserDataLoader

    loader = UserDataLoader(min_anime_count=30)
    usernames = loader.read_users_from_file("datas_user.txt")

    if not usernames:
        print("\n❌ 錯誤: 檔案中沒有使用者")
        sys.exit(1)

    await loader.load_users(usernames)

    # 關閉 client
    try:
        await loader.client.close()
    except AttributeError:
        pass

    # 最終檢查
    print("\n" + "=" * 80)
    print("📊 最終資料統計")
    print("=" * 80)

    with Session(engine) as session:
        anime_count = len(session.exec(select(BERTAnime)).all())
        user_ids = session.exec(select(BERTUserAnimeList.user_id).distinct()).all()
        user_count = len(user_ids)
        record_count = len(session.exec(select(BERTUserAnimeList)).all())

        print(f"  動畫數量: {anime_count}")
        print(f"  使用者數量: {user_count}")
        print(f"  訓練記錄: {record_count}")

        if user_count > 0:
            avg_anime = record_count / user_count
            print(f"  平均每使用者: {avg_anime:.1f} 部動畫")

    print("=" * 80)
    print("\n✅ 完成！現在可以執行 train.bat 開始訓練")


if __name__ == "__main__":
    try:
        asyncio.run(clean_and_reload())
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
