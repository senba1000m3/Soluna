"""
BERT4Rec 訓練過程視覺化工具
提供 Loss 和 Accuracy 圖表繪製功能
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """訓練過程視覺化器"""

    def __init__(self, save_dir: Path):
        """
        初始化視覺化器

        Args:
            save_dir: 圖表儲存目錄
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 設定中文字體
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YhHei",
            "SimHei",
            "Arial Unicode MS",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        # 設定樣式
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_loss(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_name: str = "loss_curve.png",
    ) -> None:
        """
        繪製 Loss 曲線

        Args:
            train_losses: 訓練 Loss 列表
            val_losses: 驗證 Loss 列表
            save_name: 儲存檔案名稱
        """
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.title("Training and Validation Loss", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12, loc="best")
        plt.grid(True, alpha=0.3)

        # 標註最小值
        min_train_idx = np.argmin(train_losses)
        min_val_idx = np.argmin(val_losses)

        plt.plot(min_train_idx + 1, train_losses[min_train_idx], "b*", markersize=15)
        plt.plot(min_val_idx + 1, val_losses[min_val_idx], "r*", markersize=15)

        plt.annotate(
            f"Min: {train_losses[min_train_idx]:.4f}",
            xy=(min_train_idx + 1, train_losses[min_train_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.3),
        )

        plt.annotate(
            f"Min: {val_losses[min_val_idx]:.4f}",
            xy=(min_val_idx + 1, val_losses[min_val_idx]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.3),
        )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  📊 Loss 曲線已儲存: {save_path}")

    def plot_accuracy(
        self,
        train_accuracies: Dict[int, List[float]],
        val_accuracies: Dict[int, List[float]],
        save_name: str = "accuracy_curve.png",
    ) -> None:
        """
        繪製 Top-K Accuracy 曲線

        Args:
            train_accuracies: 訓練準確率字典 {k: [acc1, acc2, ...]}
            val_accuracies: 驗證準確率字典 {k: [acc1, acc2, ...]}
            save_name: 儲存檔案名稱
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 訓練準確率
        ax1 = axes[0]
        for k in sorted(train_accuracies.keys()):
            epochs = range(1, len(train_accuracies[k]) + 1)
            ax1.plot(
                epochs,
                train_accuracies[k],
                label=f"Top-{k}",
                linewidth=2,
                marker="o",
                markersize=3,
            )

        ax1.set_title("Training Accuracy", fontsize=16, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=14)
        ax1.set_ylabel("Accuracy", fontsize=14)
        ax1.legend(fontsize=12, loc="best")
        ax1.grid(True, alpha=0.3)

        # 驗證準確率
        ax2 = axes[1]
        for k in sorted(val_accuracies.keys()):
            epochs = range(1, len(val_accuracies[k]) + 1)
            ax2.plot(
                epochs,
                val_accuracies[k],
                label=f"Top-{k}",
                linewidth=2,
                marker="o",
                markersize=3,
            )

        ax2.set_title("Validation Accuracy", fontsize=16, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=14)
        ax2.set_ylabel("Accuracy", fontsize=14)
        ax2.legend(fontsize=12, loc="best")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  📊 Accuracy 曲線已儲存: {save_path}")

    def plot_combined_metrics(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: Dict[int, List[float]],
        val_accuracies: Dict[int, List[float]],
        save_name: str = "combined_metrics.png",
    ) -> None:
        """
        繪製綜合指標圖（Loss + Accuracy）

        Args:
            train_losses: 訓練 Loss 列表
            val_losses: 驗證 Loss 列表
            train_accuracies: 訓練準確率字典
            val_accuracies: 驗證準確率字典
            save_name: 儲存檔案名稱
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(train_losses) + 1)

        # 1. Loss 曲線
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. 訓練準確率
        ax2 = axes[0, 1]
        for k in sorted(train_accuracies.keys()):
            ax2.plot(
                epochs,
                train_accuracies[k],
                label=f"Top-{k}",
                linewidth=2,
                marker="o",
                markersize=2,
            )
        ax2.set_title("Training Accuracy", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. 驗證準確率
        ax3 = axes[1, 0]
        for k in sorted(val_accuracies.keys()):
            ax3.plot(
                epochs,
                val_accuracies[k],
                label=f"Top-{k}",
                linewidth=2,
                marker="o",
                markersize=2,
            )
        ax3.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Accuracy", fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. 最佳 Top-K 比較
        ax4 = axes[1, 1]
        k_values = sorted(train_accuracies.keys())
        train_best = [max(train_accuracies[k]) for k in k_values]
        val_best = [max(val_accuracies[k]) for k in k_values]

        x = np.arange(len(k_values))
        width = 0.35

        ax4.bar(x - width / 2, train_best, width, label="Training", alpha=0.8)
        ax4.bar(x + width / 2, val_best, width, label="Validation", alpha=0.8)

        ax4.set_title("Best Top-K Accuracy Comparison", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Top-K", fontsize=12)
        ax4.set_ylabel("Accuracy", fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"Top-{k}" for k in k_values])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis="y")

        # 添加數值標籤
        for i, (train_acc, val_acc) in enumerate(zip(train_best, val_best)):
            ax4.text(
                i - width / 2,
                train_acc,
                f"{train_acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax4.text(
                i + width / 2,
                val_acc,
                f"{val_acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  📊 綜合指標圖已儲存: {save_path}")

    def plot_learning_curve(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_name: str = "learning_curve.png",
    ) -> None:
        """
        繪製學習曲線（包含平滑曲線）

        Args:
            train_losses: 訓練 Loss 列表
            val_losses: 驗證 Loss 列表
            save_name: 儲存檔案名稱
        """
        epochs = range(1, len(train_losses) + 1)

        # 計算移動平均
        def moving_average(data, window=10):
            return np.convolve(data, np.ones(window) / window, mode="valid")

        window_size = min(10, len(train_losses) // 10 + 1)
        train_smooth = moving_average(train_losses, window_size)
        val_smooth = moving_average(val_losses, window_size)

        plt.figure(figsize=(14, 7))

        # 原始曲線（半透明）
        plt.plot(epochs, train_losses, "b-", alpha=0.3, linewidth=1)
        plt.plot(epochs, val_losses, "r-", alpha=0.3, linewidth=1)

        # 平滑曲線
        smooth_epochs = range(1, len(train_smooth) + 1)
        plt.plot(
            smooth_epochs,
            train_smooth,
            "b-",
            label="Training Loss (smoothed)",
            linewidth=2.5,
        )
        plt.plot(
            smooth_epochs,
            val_smooth,
            "r-",
            label="Validation Loss (smoothed)",
            linewidth=2.5,
        )

        plt.title("Learning Curve (Smoothed)", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12, loc="best")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  📊 學習曲線已儲存: {save_path}")

    def save_metrics_json(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: Dict[int, List[float]],
        val_accuracies: Dict[int, List[float]],
        save_name: str = "training_metrics.json",
    ) -> None:
        """
        儲存訓練指標為 JSON 格式

        Args:
            train_losses: 訓練 Loss 列表
            val_losses: 驗證 Loss 列表
            train_accuracies: 訓練準確率字典
            val_accuracies: 驗證準確率字典
            save_name: 儲存檔案名稱
        """
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": {str(k): v for k, v in train_accuracies.items()},
            "val_accuracies": {str(k): v for k, v in val_accuracies.items()},
            "summary": {
                "best_train_loss": float(min(train_losses)),
                "best_val_loss": float(min(val_losses)),
                "final_train_loss": float(train_losses[-1]),
                "final_val_loss": float(val_losses[-1]),
                "best_train_acc": {
                    k: float(max(v)) for k, v in train_accuracies.items()
                },
                "best_val_acc": {k: float(max(v)) for k, v in val_accuracies.items()},
            },
        }

        save_path = self.save_dir / save_name
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"  💾 訓練指標已儲存: {save_path}")

    def plot_all(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: Dict[int, List[float]],
        val_accuracies: Dict[int, List[float]],
    ) -> None:
        """
        繪製所有圖表

        Args:
            train_losses: 訓練 Loss 列表
            val_losses: 驗證 Loss 列表
            train_accuracies: 訓練準確率字典
            val_accuracies: 驗證準確率字典
        """
        print("\n📊 正在生成訓練視覺化圖表...")

        self.plot_loss(train_losses, val_losses)
        self.plot_accuracy(train_accuracies, val_accuracies)
        self.plot_combined_metrics(
            train_losses, val_losses, train_accuracies, val_accuracies
        )
        self.plot_learning_curve(train_losses, val_losses)
        self.save_metrics_json(
            train_losses, val_losses, train_accuracies, val_accuracies
        )

        print("  ✅ 所有圖表生成完成！")


def plot_final_results(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: Dict[int, List[float]],
    val_accuracies: Dict[int, List[float]],
    save_dir: Path,
) -> None:
    """
    繪製最終結果（便捷函數）

    Args:
        train_losses: 訓練 Loss 列表
        val_losses: 驗證 Loss 列表
        train_accuracies: 訓練準確率字典
        val_accuracies: 驗證準確率字典
        save_dir: 儲存目錄
    """
    visualizer = TrainingVisualizer(save_dir)
    visualizer.plot_all(train_losses, val_losses, train_accuracies, val_accuracies)
