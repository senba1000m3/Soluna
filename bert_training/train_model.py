"""
BERT4Rec 完整訓練腳本
包含 Loss 和 Top-K Accuracy 計算及視覺化

使用方式:
    python train_model.py
    python train_model.py --epochs 200 --batch-size 128
    python train_model.py --gpu --fp16
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlmodel import Session, create_engine, select
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 修正 Windows 編碼問題
if sys.platform == "win32":
    import codecs

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# 導入配置和視覺化
from config import Config
from visualize import TrainingVisualizer

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 資料庫模型（簡化版，避免重複導入）
# ============================================================================

from sqlmodel import Field, SQLModel


class BERTAnime(SQLModel, table=True):
    """動畫資料表"""

    __tablename__ = "bert_anime"
    id: int = Field(primary_key=True)
    title_romaji: str
    title_english: str | None = None
    title_native: str | None = None


class BERTUserAnimeList(SQLModel, table=True):
    """使用者動畫列表"""

    __tablename__ = "bert_user_anime_list"
    id: int | None = Field(default=None, primary_key=True)
    user_id: int
    username: str
    anime_id: int
    status: str
    score: float = 0.0
    progress: int = 0


# ============================================================================
# BERT4Rec 模型
# ============================================================================


class BERT4Rec(nn.Module):
    """BERT4Rec 推薦模型"""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 200,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Token embeddings (+3 for PAD, MASK, CLS)
        self.item_embedding = nn.Embedding(num_items + 3, hidden_size, padding_idx=0)
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

        # Output layer
        self.out = nn.Linear(hidden_size, num_items + 3)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Special tokens
        self.pad_token = 0
        self.mask_token = num_items + 1
        self.cls_token = num_items + 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, seq_len = input_ids.size()

        # Embeddings
        item_emb = self.item_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        embeddings = item_emb + position_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Attention mask
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token).float()

        # Transformer
        hidden_states = self.transformer(
            embeddings, src_key_padding_mask=(attention_mask == 0)
        )

        # Output
        logits = self.out(hidden_states)
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
        self.user_sequences = user_sequences
        self.item_to_idx = item_to_idx
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.pad_token = 0

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.user_sequences[idx]
        sequence = [
            self.item_to_idx[item] for item in sequence if item in self.item_to_idx
        ]

        if not sequence:
            sequence = [self.pad_token]

        # 截斷或填充
        if len(sequence) > self.max_seq_len:
            sequence = sequence[-self.max_seq_len :]
        else:
            sequence = [self.pad_token] * (self.max_seq_len - len(sequence)) + sequence

        labels = torch.tensor(sequence, dtype=torch.long)
        input_ids = labels.clone()
        attention_mask = (input_ids != self.pad_token).long()

        # 隨機遮罩
        mask_positions = torch.rand(self.max_seq_len) < self.mask_prob
        mask_positions = mask_positions & (input_ids != self.pad_token)

        if self.mask_token is not None:
            input_ids[mask_positions] = self.mask_token

        return input_ids, labels, attention_mask


# ============================================================================
# 訓練器（含準確率計算）
# ============================================================================


class BERT4RecTrainer:
    """BERT4Rec 訓練器（含 Top-K Accuracy）"""

    def __init__(
        self,
        model: BERT4Rec,
        device: torch.device,
        learning_rate: float = 1e-3,
        use_fp16: bool = False,
        top_k_list: List[int] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_fp16 = use_fp16
        self.top_k_list = top_k_list or [1, 5, 10, 20]

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

        # FP16 training
        if use_fp16 and device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ 啟用 FP16 訓練")
        else:
            self.scaler = None

        # 記錄指標
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = {k: [] for k in self.top_k_list}
        self.val_accuracies = {k: [] for k in self.top_k_list}

    def calculate_top_k_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[int, float]:
        """
        計算 Top-K 準確率

        Args:
            logits: [batch_size, seq_len, num_items]
            labels: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            {k: accuracy} 字典
        """
        batch_size, seq_len, num_items = logits.size()

        # 只計算非 padding 位置
        mask = attention_mask.bool()

        # 獲取預測的 top-k 項目
        _, top_k_preds = torch.topk(logits, max(self.top_k_list), dim=-1)

        accuracies = {}
        for k in self.top_k_list:
            # 檢查真實標籤是否在 top-k 預測中
            top_k = top_k_preds[:, :, :k]  # [batch_size, seq_len, k]
            labels_expanded = labels.unsqueeze(-1).expand_as(
                top_k
            )  # [batch_size, seq_len, k]

            # 檢查是否匹配
            correct = (top_k == labels_expanded).any(dim=-1)  # [batch_size, seq_len]

            # 只計算有效位置
            correct = correct & mask
            accuracy = correct.sum().float() / mask.sum().float()
            accuracies[k] = accuracy.item()

        return accuracies

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[int, float]]:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_accuracies = {k: 0.0 for k in self.top_k_list}

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
                    logits_2d = logits.view(-1, logits.size(-1))
                    labels_1d = labels.view(-1)
                    loss = self.criterion(logits_2d, labels_1d)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                logits_2d = logits.view(-1, logits.size(-1))
                labels_1d = labels.view(-1)
                loss = self.criterion(logits_2d, labels_1d)

                loss.backward()
                self.optimizer.step()

            # 計算準確率
            with torch.no_grad():
                batch_acc = self.calculate_top_k_accuracy(
                    logits, labels, attention_mask
                )
                for k in self.top_k_list:
                    epoch_accuracies[k] += batch_acc[k]

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "top1_acc": f"{batch_acc[1]:.4f}"}
            )

        avg_loss = total_loss / num_batches
        avg_accuracies = {k: v / num_batches for k, v in epoch_accuracies.items()}

        self.train_losses.append(avg_loss)
        for k in self.top_k_list:
            self.train_accuracies[k].append(avg_accuracies[k])

        return avg_loss, avg_accuracies

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[int, float]]:
        """驗證"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        epoch_accuracies = {k: 0.0 for k in self.top_k_list}

        for input_ids, labels, attention_mask in tqdm(
            dataloader, desc="驗證中", unit="batch"
        ):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            logits = self.model(input_ids, attention_mask)
            logits_2d = logits.view(-1, logits.size(-1))
            labels_1d = labels.view(-1)
            loss = self.criterion(logits_2d, labels_1d)

            # 計算準確率
            batch_acc = self.calculate_top_k_accuracy(logits, labels, attention_mask)
            for k in self.top_k_list:
                epoch_accuracies[k] += batch_acc[k]

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracies = {k: v / num_batches for k, v in epoch_accuracies.items()}

        self.val_losses.append(avg_loss)
        for k in self.top_k_list:
            self.val_accuracies[k].append(avg_accuracies[k])

        return avg_loss, avg_accuracies

    def save_checkpoint(self, epoch: int, filepath: Path) -> None:
        """儲存 checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "model_config": {
                "num_items": self.model.num_items,
                "max_seq_len": self.model.max_seq_len,
                "hidden_size": self.model.hidden_size,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint 已儲存: {filepath}")


# ============================================================================
# 資料載入
# ============================================================================


def load_dataset_from_db(db_path: Path) -> Tuple[List[List[int]], Dict, Dict, int]:
    """從資料庫載入資料集"""
    print("\n📚 載入資料集...")

    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # 載入所有動畫
        animes = session.exec(select(BERTAnime)).all()
        anime_ids = sorted([anime.id for anime in animes])

        # 建立映射
        item_to_idx = {anime_id: idx + 1 for idx, anime_id in enumerate(anime_ids)}
        idx_to_item = {idx: anime_id for anime_id, idx in item_to_idx.items()}
        num_items = len(anime_ids)

        print(f"  ✓ 載入 {num_items} 部動畫")

        # 載入使用者序列
        user_lists = session.exec(select(BERTUserAnimeList)).all()

        # 按使用者分組
        user_sequences_dict = {}
        for entry in user_lists:
            if entry.user_id not in user_sequences_dict:
                user_sequences_dict[entry.user_id] = []
            user_sequences_dict[entry.user_id].append(entry.anime_id)

        user_sequences = list(user_sequences_dict.values())
        print(f"  ✓ 載入 {len(user_sequences)} 個使用者序列")

    return user_sequences, item_to_idx, idx_to_item, num_items


def split_dataset(
    user_sequences: List[List[int]], val_ratio: float = 0.1
) -> Tuple[List[List[int]], List[List[int]]]:
    """分割訓練集和驗證集"""
    num_val = int(len(user_sequences) * val_ratio)
    indices = np.random.permutation(len(user_sequences))

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_sequences = [user_sequences[i] for i in train_indices]
    val_sequences = [user_sequences[i] for i in val_indices]

    return train_sequences, val_sequences


# ============================================================================
# 主訓練流程
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="訓練 BERT4Rec 推薦模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 訓練參數
    parser.add_argument("--epochs", type=int, default=200, help="訓練輪數 (預設: 200)")
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
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設: 42)")

    args = parser.parse_args()

    # 初始化配置
    config = Config()
    config.update_from_args(args)
    config.print_config()

    # 設定隨機種子
    torch.manual_seed(config.training.random_seed)
    np.random.seed(config.training.random_seed)

    # 設定裝置
    if config.training.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\n⚠️  使用 CPU")

    try:
        # 載入資料
        user_sequences, item_to_idx, idx_to_item, num_items = load_dataset_from_db(
            config.paths.db_path
        )

        if len(user_sequences) == 0:
            print("\n❌ 錯誤: 沒有可用的訓練資料")
            print("請先執行準備資料的步驟")
            sys.exit(1)

        # 分割資料
        train_sequences, val_sequences = split_dataset(
            user_sequences, val_ratio=config.training.val_ratio
        )
        print(f"\n  訓練集: {len(train_sequences)} 個序列")
        print(f"  驗證集: {len(val_sequences)} 個序列")

        # 建立資料集
        mask_token = num_items + 1
        train_dataset = BERT4RecDataset(
            train_sequences,
            item_to_idx,
            config.training.max_seq_len,
            mask_token=mask_token,
        )
        val_dataset = BERT4RecDataset(
            val_sequences,
            item_to_idx,
            config.training.max_seq_len,
            mask_token=mask_token,
        )

        # 建立 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )

        # 建立模型
        model = BERT4Rec(
            num_items=num_items,
            max_seq_len=config.training.max_seq_len,
            hidden_size=config.training.hidden_size,
            num_layers=config.training.num_layers,
            num_heads=config.training.num_heads,
            dropout=config.training.dropout,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ 模型建立完成")
        print(f"  總參數量: {total_params:,}")

        # 建立訓練器
        trainer = BERT4RecTrainer(
            model,
            device,
            learning_rate=config.training.learning_rate,
            use_fp16=config.training.use_fp16 and config.training.use_gpu,
            top_k_list=config.data.top_k_list,
        )

        # 初始化視覺化器
        visualizer = TrainingVisualizer(config.paths.plot_dir)

        # 訓練
        print("\n" + "=" * 80)
        print("🎯 開始訓練")
        print("=" * 80)

        best_val_loss = float("inf")
        best_val_acc = 0.0

        for epoch in range(1, config.training.epochs + 1):
            print(f"\n📋 Epoch {epoch}/{config.training.epochs}")
            print("-" * 80)

            # 訓練
            train_loss, train_acc = trainer.train_epoch(train_loader)
            print(f"  訓練 Loss: {train_loss:.4f}")
            print(f"  訓練 Top-1 Acc: {train_acc[1]:.4f}")
            print(f"  訓練 Top-5 Acc: {train_acc[5]:.4f}")
            print(f"  訓練 Top-10 Acc: {train_acc[10]:.4f}")

            # 驗證
            val_loss, val_acc = trainer.validate(val_loader)
            print(f"  驗證 Loss: {val_loss:.4f}")
            print(f"  驗證 Top-1 Acc: {val_acc[1]:.4f}")
            print(f"  驗證 Top-5 Acc: {val_acc[5]:.4f}")
            print(f"  驗證 Top-10 Acc: {val_acc[10]:.4f}")

            # 儲存 checkpoint
            if epoch % config.training.save_every_n_epochs == 0:
                checkpoint_path = (
                    config.paths.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                )
                trainer.save_checkpoint(epoch, checkpoint_path)

            # 儲存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc[1]
                best_model_path = config.paths.model_dir / "best_model.pth"
                trainer.save_checkpoint(epoch, best_model_path)
                print(f"  🌟 新的最佳模型！")

        # 訓練完成
        print("\n" + "=" * 80)
        print("🎉 訓練完成！")
        print("=" * 80)
        print(f"  最佳驗證 Loss: {best_val_loss:.4f}")
        print(f"  最佳驗證 Top-1 Acc: {best_val_acc:.4f}")

        # 儲存最終模型
        final_model_path = config.paths.model_dir / "final_model.pth"
        trainer.save_checkpoint(config.training.epochs, final_model_path)

        # 儲存映射資料
        mapping_path = config.paths.model_dir / "item_mappings.pkl"
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

        # 儲存配置
        config_path = config.paths.model_dir / "training_config.json"
        config.save_to_file(config_path)
        print(f"  配置已儲存: {config_path}")

        # 生成視覺化圖表
        visualizer.plot_all(
            trainer.train_losses,
            trainer.val_losses,
            trainer.train_accuracies,
            trainer.val_accuracies,
        )

        print("\n✅ 所有輸出已儲存至:")
        print(f"  模型: {config.paths.model_dir}")
        print(f"  圖表: {config.paths.plot_dir}")
        print(f"  日誌: {config.paths.log_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  訓練被中斷")
        sys.exit(130)
    except Exception as e:
        logger.error(f"訓練失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
