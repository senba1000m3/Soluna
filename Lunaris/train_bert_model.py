"""
BERT4Rec 模型訓練腳本
使用準備好的資料集訓練推薦模型

使用方式:
    python train_bert_model.py
    python train_bert_model.py --epochs 50 --batch-size 128
    python train_bert_model.py --gpu --fp16
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlmodel import Session, create_engine, select
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Fix Windows encoding issue
if sys.platform == "win32":
    import codecs

    # Check if stdout/stderr have buffer attribute (not in uv run)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_bert_model.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 資料庫路徑
BERT_DB_PATH = "bert.db"
BERT_DB_URL = f"sqlite:///{BERT_DB_PATH}"

# 模型輸出目錄
OUTPUT_DIR = Path("bert_models")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 載入資料庫模型
# ============================================================================

from prepare_bert_dataset import BERTAnime, BERTUserAnimeList

# ============================================================================
# BERT4Rec 模型架構
# ============================================================================


class BERT4Rec(nn.Module):
    """
    BERT4Rec 模型實現
    基於 Transformer Encoder 的序列推薦模型
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 200,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        初始化 BERT4Rec 模型

        Args:
            num_items: 動畫總數
            max_seq_len: 最大序列長度
            hidden_size: 隱藏層維度
            num_layers: Transformer 層數
            num_heads: 注意力頭數
            dropout: Dropout 比率
        """
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Token embeddings
        # +3 for [PAD], [MASK], [CLS] tokens
        self.item_embedding = nn.Embedding(num_items + 3, hidden_size, padding_idx=0)

        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer (num_items + 3 for PAD, MASK, CLS)
        self.out = nn.Linear(hidden_size, num_items + 3)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Special tokens
        self.pad_token = 0
        self.mask_token = num_items + 1
        self.cls_token = num_items + 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        前向傳播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, num_items]
        """
        batch_size, seq_len = input_ids.size()

        # Item embeddings
        item_emb = self.item_embedding(input_ids)  # [B, L, H]

        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)  # [B, L, H]

        # Combine embeddings
        embeddings = item_emb + position_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Create attention mask for padding
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token).float()

        # Transformer expects mask of shape [batch, seq_len]
        # with 1 for tokens to attend to, 0 for padding
        # We need to invert it: 1 -> 0, 0 -> -inf
        extended_attention_mask = (1.0 - attention_mask) * -10000.0

        # Transformer encoding
        hidden_states = self.transformer(
            embeddings, src_key_padding_mask=(attention_mask == 0)
        )

        # Output projection
        logits = self.out(hidden_states)  # [B, L, num_items]

        return logits


# ============================================================================
# 資料集
# ============================================================================


class BERT4RecDataset(Dataset):
    """BERT4Rec 訓練資料集"""

    def __init__(
        self,
        user_sequences: List[List[int]],
        item_to_idx: Dict[int, int],
        max_seq_len: int = 200,
        mask_prob: float = 0.15,
        mask_token: int = None,
    ):
        """
        初始化資料集

        Args:
            user_sequences: 使用者觀看序列列表
            item_to_idx: 動畫 ID 到索引的映射
            max_seq_len: 最大序列長度
            mask_prob: 遮罩概率
            mask_token: 遮罩 token ID
        """
        self.user_sequences = user_sequences
        self.item_to_idx = item_to_idx
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = len(item_to_idx)
        self.pad_token = 0

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        取得訓練樣本

        Returns:
            input_ids: 遮罩後的序列
            labels: 原始序列
            attention_mask: 注意力遮罩
        """
        # 取得序列並轉換為索引
        sequence = self.user_sequences[idx]
        # 只保留有效的動畫 ID（在映射中的）
        sequence = [
            self.item_to_idx[item] for item in sequence if item in self.item_to_idx
        ]

        # 如果過濾後序列為空，使用 padding
        if not sequence:
            sequence = [self.pad_token]

        # 截斷或填充到 max_seq_len
        if len(sequence) > self.max_seq_len:
            sequence = sequence[-self.max_seq_len :]
        else:
            sequence = [self.pad_token] * (self.max_seq_len - len(sequence)) + sequence

        # 轉換為 tensor
        labels = torch.tensor(sequence, dtype=torch.long)

        # 建立遮罩序列
        input_ids = labels.clone()
        attention_mask = (input_ids != self.pad_token).long()

        # 隨機遮罩部分 token
        mask_positions = torch.rand(self.max_seq_len) < self.mask_prob
        # 不遮罩 padding
        mask_positions = mask_positions & (input_ids != self.pad_token)

        if self.mask_token is not None:
            input_ids[mask_positions] = self.mask_token

        return input_ids, labels, attention_mask


# ============================================================================
# 訓練器
# ============================================================================


class BERT4RecTrainer:
    """BERT4Rec 訓練器"""

    def __init__(
        self,
        model: BERT4Rec,
        device: torch.device,
        learning_rate: float = 1e-3,
        use_fp16: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_fp16 = use_fp16

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, reduction="mean"
        )  # 忽略 padding

        # FP16 training
        if use_fp16 and device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ 啟用 FP16 訓練")
        else:
            self.scaler = None

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="訓練中", unit="batch")
        for input_ids, labels, attention_mask in progress_bar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.use_fp16 and self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    # Reshape for CrossEntropyLoss
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                    loss = self.criterion(logits, labels)

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """驗證"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for input_ids, labels, attention_mask in tqdm(
            dataloader, desc="驗證中", unit="batch"
        ):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            logits = self.model(input_ids, attention_mask)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, epoch: int, filepath: Path) -> None:
        """儲存 checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "model_config": {
                "num_items": self.model.num_items,
                "max_seq_len": self.model.max_seq_len,
                "hidden_size": self.model.hidden_size,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint 已儲存: {filepath}")


# ============================================================================
# 資料載入與預處理
# ============================================================================


def load_dataset_from_db() -> Tuple[List[List[int]], Dict[int, int], int]:
    """
    從資料庫載入資料集

    Returns:
        user_sequences: 使用者序列列表
        item_to_idx: 動畫 ID 到索引的映射
        num_items: 動畫總數
    """
    print("\n" + "=" * 80)
    print("📚 載入資料集")
    print("=" * 80)

    engine = create_engine(BERT_DB_URL, echo=False)

    with Session(engine) as session:
        # 載入所有動畫
        animes = session.exec(select(BERTAnime)).all()
        num_items = len(animes)
        print(f"  ✓ 動畫總數: {num_items}")

        # 建立 ID 映射 (AniList ID -> 模型索引)
        # 保留 0 給 padding, num_items+1 給 mask, num_items+2 給 cls
        item_to_idx = {anime.id: idx + 1 for idx, anime in enumerate(animes)}
        idx_to_item = {idx + 1: anime.id for idx, anime in enumerate(animes)}

        print(f"  ✓ ID 映射建立完成")

        # 載入使用者序列
        user_lists = session.exec(select(BERTUserAnimeList)).all()
        print(f"  ✓ 使用者列表記錄: {len(user_lists)}")

        # 按使用者分組
        user_sequences_dict = {}
        skipped_count = 0
        for entry in user_lists:
            user_id = entry.user_id
            anime_id = entry.anime_id

            # 使用所有狀態的動畫（COMPLETED, CURRENT, PLANNING, DROPPED, PAUSED 等）
            # 不過濾狀態，因為使用者的觀看記錄都有參考價值

            # 檢查動畫 ID 是否在映射中
            if anime_id not in item_to_idx:
                skipped_count += 1
                continue

            if user_id not in user_sequences_dict:
                user_sequences_dict[user_id] = []

            user_sequences_dict[user_id].append(anime_id)

        if skipped_count > 0:
            print(f"  ⚠️  跳過 {skipped_count} 個不在資料庫中的動畫")

        # 轉換為列表
        user_sequences = list(user_sequences_dict.values())

        # 過濾太短的序列 (至少 5 部)
        user_sequences = [seq for seq in user_sequences if len(seq) >= 5]

        print(f"  ✓ 有效使用者數: {len(user_sequences)}")
        print(f"  ✓ 平均序列長度: {np.mean([len(seq) for seq in user_sequences]):.1f}")
        print(f"  ✓ 最長序列: {max(len(seq) for seq in user_sequences)}")
        print(f"  ✓ 最短序列: {min(len(seq) for seq in user_sequences)}")

    return user_sequences, item_to_idx, idx_to_item, num_items


def split_dataset(
    user_sequences: List[List[int]], val_ratio: float = 0.1
) -> Tuple[List[List[int]], List[List[int]]]:
    """分割訓練集和驗證集"""
    num_val = int(len(user_sequences) * val_ratio)
    num_train = len(user_sequences) - num_val

    # 隨機打亂
    indices = np.random.permutation(len(user_sequences))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_sequences = [user_sequences[i] for i in train_indices]
    val_sequences = [user_sequences[i] for i in val_indices]

    return train_sequences, val_sequences


# ============================================================================
# 主程式
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="訓練 BERT4Rec 推薦模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本訓練
  python train_bert_model.py

  # 使用 GPU 和 FP16
  python train_bert_model.py --gpu --fp16

  # 自訂參數
  python train_bert_model.py --epochs 50 --batch-size 128 --lr 0.001

  # 從 checkpoint 繼續訓練
  python train_bert_model.py --resume bert_models/checkpoint_epoch_10.pth
        """,
    )

    # 訓練參數
    parser.add_argument("--epochs", type=int, default=30, help="訓練輪數 (預設: 30)")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="批次大小 (預設: 64)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率 (預設: 0.001)")
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="驗證集比例 (預設: 0.1)"
    )

    # 模型參數
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="隱藏層大小 (預設: 256)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Transformer 層數 (預設: 2)"
    )
    parser.add_argument("--num-heads", type=int, default=4, help="注意力頭數 (預設: 4)")
    parser.add_argument(
        "--max-seq-len", type=int, default=200, help="最大序列長度 (預設: 200)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout (預設: 0.1)"
    )

    # 硬體
    parser.add_argument("--gpu", action="store_true", help="使用 GPU")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 訓練")

    # 其他
    parser.add_argument("--resume", type=str, help="從 checkpoint 繼續訓練")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設: 42)")

    args = parser.parse_args()

    # 設定隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 設定裝置
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  使用 CPU (建議使用 GPU 以加速訓練)")

    print("\n" + "=" * 80)
    print("🚀 BERT4Rec 模型訓練")
    print("=" * 80)
    print(f"  訓練輪數: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  學習率: {args.lr}")
    print(f"  最大序列長度: {args.max_seq_len}")
    print(f"  隱藏層大小: {args.hidden_size}")
    print(f"  Transformer 層數: {args.num_layers}")
    print(f"  注意力頭數: {args.num_heads}")
    print(f"  FP16: {'啟用' if args.fp16 else '停用'}")
    print("=" * 80)

    try:
        # 載入資料
        user_sequences, item_to_idx, idx_to_item, num_items = load_dataset_from_db()

        if len(user_sequences) == 0:
            print("\n❌ 錯誤: 沒有可用的訓練資料")
            print("請先執行: python prepare_bert_dataset.py --users USERNAME")
            sys.exit(1)

        # 分割資料
        train_sequences, val_sequences = split_dataset(
            user_sequences, val_ratio=args.val_ratio
        )
        print(f"\n  訓練集: {len(train_sequences)} 個序列")
        print(f"  驗證集: {len(val_sequences)} 個序列")

        # 建立資料集
        mask_token = num_items + 1
        train_dataset = BERT4RecDataset(
            train_sequences, item_to_idx, args.max_seq_len, mask_token=mask_token
        )
        val_dataset = BERT4RecDataset(
            val_sequences, item_to_idx, args.max_seq_len, mask_token=mask_token
        )

        # 建立 DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # 建立模型
        model = BERT4Rec(
            num_items=num_items,
            max_seq_len=args.max_seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

        print(f"\n✅ 模型建立完成")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  總參數量: {total_params:,}")

        # 建立訓練器
        trainer = BERT4RecTrainer(
            model, device, learning_rate=args.lr, use_fp16=args.fp16 and args.gpu
        )

        # 訓練
        print("\n" + "=" * 80)
        print("🎯 開始訓練")
        print("=" * 80)

        best_val_loss = float("inf")
        start_epoch = 1

        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n📋 Epoch {epoch}/{args.epochs}")
            print("-" * 80)

            # 訓練
            train_loss = trainer.train_epoch(train_loader)
            print(f"  訓練 Loss: {train_loss:.4f}")

            # 驗證
            val_loss = trainer.validate(val_loader)
            print(f"  驗證 Loss: {val_loss:.4f}")

            # 儲存 checkpoint
            if epoch % 5 == 0 or epoch == args.epochs:
                checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pth"
                trainer.save_checkpoint(epoch, checkpoint_path)

            # 儲存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = OUTPUT_DIR / "best_model.pth"
                trainer.save_checkpoint(epoch, best_model_path)
                print(f"  🌟 新的最佳模型！驗證 Loss: {val_loss:.4f}")

        # 訓練完成
        print("\n" + "=" * 80)
        print("🎉 訓練完成！")
        print("=" * 80)
        print(f"  最佳驗證 Loss: {best_val_loss:.4f}")
        print(f"  模型已儲存至: {OUTPUT_DIR}")

        # 儲存映射資料
        mapping_path = OUTPUT_DIR / "item_mappings.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(
                {
                    "item_to_idx": item_to_idx,
                    "idx_to_item": idx_to_item,
                    "num_items": num_items,
                },
                f,
            )
        print(f"  映射資料已儲存: {mapping_path}")

        print("\n📝 下一步:")
        print("  1. 測試模型: python test_bert_model.py")
        print("  2. 整合到推薦系統: 修改 hybrid_recommendation_engine.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  訓練被中斷")
        sys.exit(130)
    except Exception as e:
        logger.error(f"訓練失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
