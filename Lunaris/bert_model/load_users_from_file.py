"""
從檔案讀取使用者名稱並抓取其動畫列表到 BERT 資料庫

使用方式：
    python load_users_from_file.py
    python load_users_from_file.py --file custom_users.txt
    python load_users_from_file.py --min-anime 50
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set

from sqlmodel import Session, create_engine, select
from tqdm import tqdm

from anilist_client import AniListClient
from prepare_bert_dataset import BERTAnime, BERTUserAnimeList, init_bert_db

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("load_users.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

BERT_DB_URL = "sqlite:///bert.db"


class UserDataLoader:
    """從檔案載入使用者資料的工具"""

    def __init__(self, min_anime_count: int = 20):
        """
        初始化

        Args:
            min_anime_count: 使用者至少要有幾部動畫才會被加入
        """
        self.client = AniListClient()
        self.min_anime_count = min_anime_count
        self.stats = {
            "total_users": 0,
            "valid_users": 0,
            "skipped_users": 0,
            "failed_users": 0,
            "total_anime_records": 0,
            "errors": 0,
        }

    def read_users_from_file(self, file_path: str) -> List[str]:
        """
        從檔案讀取使用者名稱

        Args:
            file_path: 檔案路徑

        Returns:
            使用者名稱列表
        """
        path = Path(file_path)

        if not path.exists():
            print(f"❌ 錯誤: 檔案不存在 - {file_path}")
            sys.exit(1)

        print(f"\n📖 讀取使用者列表: {file_path}")

        usernames = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                username = line.strip()
                if username and not username.startswith("#"):
                    usernames.append(username)

        print(f"  ✓ 找到 {len(usernames)} 個使用者")
        return usernames

    async def fetch_user_anime_list(
        self, username: str
    ) -> tuple[int, List[dict]] | tuple[None, None]:
        """
        抓取使用者的動畫列表

        Args:
            username: 使用者名稱

        Returns:
            (user_id, anime_list) 或 (None, None)
        """
        try:
            # 先取得使用者資料
            user_data = await self.client.get_user_profile(username)
            if not user_data:
                logger.warning(f"無法取得使用者資料: {username}")
                return None, None

            user_id = user_data.get("id")
            if not user_id:
                logger.warning(f"使用者資料中沒有 ID: {username}")
                return None, None

            # 取得動畫列表
            anime_list = await self.client.get_user_anime_list(username)
            if not anime_list:
                logger.warning(f"使用者動畫列表為空: {username}")
                return None, None

            return user_id, anime_list

        except Exception as e:
            logger.error(f"抓取使用者 {username} 的動畫列表時發生錯誤: {e}")
            return None, None

    def store_user_anime_list(
        self,
        user_id: int,
        username: str,
        anime_list: List[dict],
        session: Session,
        anime_id_set: Set[int],
    ) -> int:
        """
        儲存使用者的動畫列表到資料庫

        Args:
            user_id: 使用者 ID
            username: 使用者名稱
            anime_list: 動畫列表
            session: 資料庫 session
            anime_id_set: 已存在的動畫 ID 集合

        Returns:
            成功儲存的記錄數
        """
        stored_count = 0

        for entry in anime_list:
            try:
                anime = entry.get("media", entry)
                anime_id = anime.get("id")

                if not anime_id or anime_id not in anime_id_set:
                    continue

                status = entry.get("status", "CURRENT")
                score = entry.get("score", 0.0)
                progress = entry.get("progress", 0)

                # 檢查是否已存在
                statement = select(BERTUserAnimeList).where(
                    BERTUserAnimeList.user_id == user_id,
                    BERTUserAnimeList.anime_id == anime_id,
                )
                existing = session.exec(statement).first()

                if not existing:
                    # 新增記錄
                    user_anime = BERTUserAnimeList(
                        user_id=user_id,
                        username=username,
                        anime_id=anime_id,
                        status=status,
                        score=score,
                        progress=progress,
                    )
                    session.add(user_anime)
                    stored_count += 1
                else:
                    # 更新現有記錄
                    existing.status = status
                    existing.score = score
                    existing.progress = progress
                    existing.updated_at = datetime.utcnow()
                    stored_count += 1

            except Exception as e:
                logger.error(f"儲存動畫記錄時發生錯誤: {e}")
                continue

        return stored_count

    async def process_user(
        self,
        username: str,
        session: Session,
        anime_id_set: Set[int],
        progress_bar: tqdm = None,
    ) -> bool:
        """
        處理單一使用者

        Args:
            username: 使用者名稱
            session: 資料庫 session
            anime_id_set: 已存在的動畫 ID 集合
            progress_bar: 進度條

        Returns:
            是否成功
        """
        try:
            if progress_bar:
                progress_bar.set_description(f"處理: {username}")

            # 抓取動畫列表
            user_id, anime_list = await self.fetch_user_anime_list(username)

            if not user_id or not anime_list:
                self.stats["failed_users"] += 1
                if progress_bar:
                    progress_bar.write(f"  ❌ {username}: 無法取得資料")
                return False

            # 檢查動畫數量
            if len(anime_list) < self.min_anime_count:
                self.stats["skipped_users"] += 1
                if progress_bar:
                    progress_bar.write(
                        f"  ⚠️  {username}: 動畫數量不足 ({len(anime_list)} < {self.min_anime_count})"
                    )
                return False

            # 儲存到資料庫
            stored_count = self.store_user_anime_list(
                user_id, username, anime_list, session, anime_id_set
            )

            if stored_count > 0:
                self.stats["valid_users"] += 1
                self.stats["total_anime_records"] += stored_count
                if progress_bar:
                    progress_bar.write(
                        f"  ✓ {username}: {len(anime_list)} 部動畫, 儲存 {stored_count} 筆"
                    )
                return True
            else:
                self.stats["skipped_users"] += 1
                if progress_bar:
                    progress_bar.write(f"  ⚠️  {username}: 沒有有效的動畫記錄")
                return False

        except Exception as e:
            logger.error(f"處理使用者 {username} 時發生錯誤: {e}")
            self.stats["errors"] += 1
            if progress_bar:
                progress_bar.write(f"  ❌ {username}: {str(e)}")
            return False

    async def load_users(self, usernames: List[str]) -> None:
        """
        載入使用者資料

        Args:
            usernames: 使用者名稱列表
        """
        print("\n" + "=" * 80)
        print("🚀 開始載入使用者資料")
        print("=" * 80)
        print(f"  使用者數量: {len(usernames)}")
        print(f"  最少動畫數: {self.min_anime_count}")
        print("=" * 80)

        self.stats["total_users"] = len(usernames)

        # 初始化資料庫
        init_bert_db()
        engine = create_engine(BERT_DB_URL, echo=False)

        with Session(engine) as session:
            # 取得現有動畫 ID
            print("\n📚 載入動畫資料...")
            animes = session.exec(select(BERTAnime)).all()
            anime_id_set = {anime.id for anime in animes}
            print(f"  ✓ 資料庫中有 {len(anime_id_set)} 部動畫")

            if len(anime_id_set) == 0:
                print("\n❌ 錯誤: 資料庫中沒有動畫資料")
                print("   請先執行: python prepare_bert_dataset.py --count 3000")
                sys.exit(1)

            # 處理每個使用者
            print(f"\n處理使用者...")
            with tqdm(total=len(usernames), unit="user") as pbar:
                for i, username in enumerate(usernames):
                    await self.process_user(username, session, anime_id_set, pbar)

                    # 每 5 個使用者 commit 一次
                    if (i + 1) % 5 == 0:
                        session.commit()
                        pbar.write(f"\n  💾 已儲存進度 ({i + 1}/{len(usernames)})")

                    # 避免過度請求
                    await asyncio.sleep(2)

                    pbar.update(1)

                # commit 最後的變更
                session.commit()

        # 列印最終統計
        self.print_stats()

    def print_stats(self) -> None:
        """列印統計資訊"""
        print("\n" + "=" * 80)
        print("📊 最終統計")
        print("=" * 80)
        print(f"  總使用者數: {self.stats['total_users']}")
        print(f"  ✅ 有效使用者: {self.stats['valid_users']}")
        print(f"  ⚠️  跳過使用者: {self.stats['skipped_users']}")
        print(f"  ❌ 失敗使用者: {self.stats['failed_users']}")
        print(f"  📝 總動畫記錄: {self.stats['total_anime_records']}")
        print(f"  ⚡ 錯誤次數: {self.stats['errors']}")

        if self.stats["valid_users"] > 0:
            avg_anime = self.stats["total_anime_records"] / self.stats["valid_users"]
            print(f"  📈 平均每使用者: {avg_anime:.1f} 部動畫")

        if self.stats["total_users"] > 0:
            success_rate = (self.stats["valid_users"] / self.stats["total_users"]) * 100
            print(f"  🎯 成功率: {success_rate:.1f}%")

        print("=" * 80)

        logger.info(
            f"載入完成: {self.stats['valid_users']} 個有效使用者, "
            f"{self.stats['total_anime_records']} 筆動畫記錄"
        )


async def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="從檔案讀取使用者並載入動畫列表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 使用預設檔案 (datas_user.txt)
  python load_users_from_file.py

  # 使用自訂檔案
  python load_users_from_file.py --file my_users.txt

  # 設定最少動畫數
  python load_users_from_file.py --min-anime 50

  # 組合使用
  python load_users_from_file.py --file users.txt --min-anime 30
        """,
    )

    parser.add_argument(
        "--file",
        type=str,
        default="datas_user.txt",
        help="使用者名稱檔案路徑 (預設 datas_user.txt)",
    )

    parser.add_argument(
        "--min-anime",
        type=int,
        default=20,
        help="使用者至少要有幾部動畫 (預設 20)",
    )

    args = parser.parse_args()

    # 建立 loader
    loader = UserDataLoader(min_anime_count=args.min_anime)

    try:
        # 讀取使用者列表
        usernames = loader.read_users_from_file(args.file)

        if not usernames:
            print("\n❌ 錯誤: 檔案中沒有使用者")
            sys.exit(1)

        # 開始載入
        await loader.load_users(usernames)

        # 關閉 client
        try:
            await loader.client.close()
        except AttributeError:
            pass  # AniListClient 可能沒有 close 方法

        print("\n✅ 完成！")
        print("\n下一步:")
        print("  1. 檢查資料庫: bert_model/bert.db")
        print("  2. 訓練模型: cd bert_model && python train_bert_model.py --epochs 20")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        try:
            await loader.client.close()
        except AttributeError:
            pass
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            await loader.client.close()
        except AttributeError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
