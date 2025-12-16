"""
檢查 BERT 訓練狀態和數據的腳本
用於診斷訓練是否正確使用數據
"""

import os
from pathlib import Path

import numpy as np
import torch
from sqlmodel import Session, create_engine, select

from prepare_bert_dataset import BERTAnime, BERTUserAnimeList

BERT_DB_URL = "sqlite:///bert.db"


def check_database():
    """檢查資料庫內容"""
    print("\n" + "=" * 80)
    print("📊 檢查資料庫")
    print("=" * 80)

    engine = create_engine(BERT_DB_URL, echo=False)

    with Session(engine) as session:
        # 檢查動畫數量
        animes = session.exec(select(BERTAnime)).all()
        print(f"✓ 動畫總數: {len(animes)}")

        # 檢查使用者記錄
        user_lists = session.exec(select(BERTUserAnimeList)).all()
        print(f"✓ 使用者動畫記錄: {len(user_lists)}")

        # 統計使用者數量
        user_ids = set([entry.user_id for entry in user_lists])
        print(f"✓ 不重複使用者: {len(user_ids)}")

        # 統計每個使用者的記錄數
        user_counts = {}
        for entry in user_lists:
            user_id = entry.user_id
            user_counts[user_id] = user_counts.get(user_id, 0) + 1

        if user_counts:
            print(f"✓ 平均每位使用者記錄數: {np.mean(list(user_counts.values())):.1f}")
            print(f"✓ 最多記錄的使用者: {max(user_counts.values())} 筆")
            print(f"✓ 最少記錄的使用者: {min(user_counts.values())} 筆")

        # 檢查狀態分布
        status_counts = {}
        for entry in user_lists:
            status = entry.status or "UNKNOWN"
            status_counts[status] = status_counts.get(status, 0) + 1

        print("\n📈 狀態分布:")
        for status, count in sorted(
            status_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(user_lists)) * 100
            print(f"  {status:15s}: {count:5d} ({percentage:5.1f}%)")

    return len(animes), len(user_lists), len(user_ids)


def check_training_data():
    """檢查訓練數據載入"""
    print("\n" + "=" * 80)
    print("🔍 檢查訓練數據載入")
    print("=" * 80)

    engine = create_engine(BERT_DB_URL, echo=False)

    with Session(engine) as session:
        # 載入動畫映射
        animes = session.exec(select(BERTAnime)).all()
        item_to_idx = {anime.id: idx + 1 for idx, anime in enumerate(animes)}
        print(f"✓ 動畫映射建立: {len(item_to_idx)} 個項目")

        # 載入使用者序列
        user_lists = session.exec(select(BERTUserAnimeList)).all()
        print(f"✓ 載入記錄: {len(user_lists)} 筆")

        # 按使用者分組
        user_sequences_dict = {}
        skipped_count = 0
        for entry in user_lists:
            user_id = entry.user_id
            anime_id = entry.anime_id

            # 檢查是否在映射中
            if anime_id not in item_to_idx:
                skipped_count += 1
                continue

            if user_id not in user_sequences_dict:
                user_sequences_dict[user_id] = []

            user_sequences_dict[user_id].append(anime_id)

        if skipped_count > 0:
            print(f"⚠️  跳過 {skipped_count} 個不在映射中的動畫")

        # 過濾序列
        all_sequences = list(user_sequences_dict.values())
        valid_sequences = [seq for seq in all_sequences if len(seq) >= 5]

        print(f"✓ 使用者序列: {len(all_sequences)} 個")
        print(f"✓ 有效序列 (>=5): {len(valid_sequences)} 個")

        if valid_sequences:
            seq_lengths = [len(seq) for seq in valid_sequences]
            print(f"✓ 平均序列長度: {np.mean(seq_lengths):.1f}")
            print(f"✓ 最長序列: {max(seq_lengths)}")
            print(f"✓ 最短序列: {min(seq_lengths)}")
            print(f"✓ 中位數: {np.median(seq_lengths):.1f}")

        return len(valid_sequences), seq_lengths if valid_sequences else []


def check_model_files():
    """檢查模型檔案"""
    print("\n" + "=" * 80)
    print("📁 檢查模型檔案")
    print("=" * 80)

    model_dir = Path("bert_model/trained_models")

    if not model_dir.exists():
        print("❌ 模型目錄不存在")
        return False

    # 檢查主要檔案
    files_to_check = {
        "best_model.pth": "最佳模型",
        "item_mappings.pkl": "項目映射",
    }

    found_files = []
    for filename, description in files_to_check.items():
        filepath = model_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {description}: {filename} ({size_mb:.2f} MB)")
            found_files.append(filename)
        else:
            print(f"❌ {description}: {filename} (不存在)")

    # 檢查 checkpoint 檔案
    checkpoint_files = list(model_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoint_files:
        print(f"\n✓ 找到 {len(checkpoint_files)} 個 checkpoint 檔案:")
        for cp in sorted(checkpoint_files)[-5:]:  # 顯示最後 5 個
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"  - {cp.name} ({size_mb:.2f} MB)")

    return len(found_files) == len(files_to_check)


def estimate_training_time():
    """估算訓練時間"""
    print("\n" + "=" * 80)
    print("⏱️  估算訓練時間")
    print("=" * 80)

    engine = create_engine(BERT_DB_URL, echo=False)

    with Session(engine) as session:
        animes = session.exec(select(BERTAnime)).all()
        user_lists = session.exec(select(BERTUserAnimeList)).all()

        item_to_idx = {anime.id: idx + 1 for idx, anime in enumerate(animes)}

        user_sequences_dict = {}
        for entry in user_lists:
            if entry.anime_id not in item_to_idx:
                continue
            if entry.user_id not in user_sequences_dict:
                user_sequences_dict[entry.user_id] = []
            user_sequences_dict[entry.user_id].append(entry.anime_id)

        valid_sequences = [seq for seq in user_sequences_dict.values() if len(seq) >= 5]

        if not valid_sequences:
            print("❌ 沒有有效的訓練序列")
            return

        # 不同批次大小的估算
        batch_sizes = [8, 16, 32, 64]
        epochs = 20

        print(f"訓練序列數: {len(valid_sequences)}")
        print(f"訓練輪數: {epochs}")
        print()

        for batch_size in batch_sizes:
            batches_per_epoch = (len(valid_sequences) + batch_size - 1) // batch_size

            # CPU 估算: ~0.5-1 秒/批次
            # GPU 估算: ~0.1-0.2 秒/批次
            cpu_time_per_batch = 0.75  # 秒
            gpu_time_per_batch = 0.15  # 秒

            cpu_total_seconds = batches_per_epoch * epochs * cpu_time_per_batch
            gpu_total_seconds = batches_per_epoch * epochs * gpu_time_per_batch

            print(f"批次大小 {batch_size}:")
            print(f"  - 每輪批次數: {batches_per_epoch}")
            print(
                f"  - CPU 估算時間: {cpu_total_seconds / 60:.1f} 分鐘 ({cpu_total_seconds:.0f} 秒)"
            )
            print(
                f"  - GPU 估算時間: {gpu_total_seconds / 60:.1f} 分鐘 ({gpu_total_seconds:.0f} 秒)"
            )
            print()

        print("⚠️  實際時間會因硬體、序列長度、模型大小而異")
        print("⚠️  如果訓練在 20 秒內完成，可能是:")
        print("     1. 訓練序列太少 (< 10)")
        print("     2. 批次大小過大 (批次數太少)")
        print("     3. 沒有正確載入數據")


def main():
    print("\n" + "=" * 80)
    print("🔬 BERT 訓練狀態檢查")
    print("=" * 80)

    # 1. 檢查資料庫
    num_animes, num_records, num_users = check_database()

    # 2. 檢查訓練數據載入
    num_sequences, seq_lengths = check_training_data()

    # 3. 檢查模型檔案
    has_model = check_model_files()

    # 4. 估算訓練時間
    estimate_training_time()

    # 總結
    print("\n" + "=" * 80)
    print("📋 總結")
    print("=" * 80)

    if num_users < 10:
        print("⚠️  警告: 使用者數量過少 (< 10)，建議至少 30 位使用者")

    if num_sequences < 10:
        print("⚠️  警告: 訓練序列過少 (< 10)，這會導致訓練極快完成")
        print("   建議: 載入更多使用者數據")

    if num_sequences >= 30:
        print("✅ 訓練序列數量充足")

    if not has_model:
        print("⚠️  模型檔案不完整，建議重新訓練")
    else:
        print("✅ 模型檔案完整")

    # 診斷 20 秒問題
    print("\n" + "=" * 80)
    print("🔍 診斷: 為什麼訓練只需要 20 秒?")
    print("=" * 80)

    if num_sequences < 10:
        print("❌ 原因: 訓練序列太少!")
        print(f"   目前只有 {num_sequences} 個序列")
        print("   解決方案: 使用 load_users_from_file.py 載入更多使用者")
    elif num_sequences < 100:
        print("⚠️  可能原因: 訓練序列較少")
        print(f"   目前有 {num_sequences} 個序列")
        print("   這是正常的，對於小數據集，訓練確實會比較快")
        print("   建議: 降低批次大小 (例如 8 或 16) 來增加訓練步數")
    else:
        print("✅ 訓練序列數量充足")
        print("   如果仍然很快完成，檢查:")
        print("   1. 批次大小是否過大")
        print("   2. 是否使用了 GPU")
        print("   3. 檢查訓練日誌中的批次數")

    print("\n✅ 檢查完成!")


if __name__ == "__main__":
    main()
