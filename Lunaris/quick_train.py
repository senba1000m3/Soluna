"""
快速訓練腳本 - 一鍵啟動 BERT4Rec 訓練

使用方式：
    python quick_train.py                    # 使用預設參數
    python quick_train.py --quick            # 快速測試（10 輪）
    python quick_train.py --full             # 完整訓練（50 輪）
    python quick_train.py --custom           # 自訂參數
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from sqlmodel import Session, create_engine, select

from prepare_bert_dataset import BERTAnime, BERTUserAnimeList

BERT_DB_URL = "sqlite:///bert.db"


def print_banner():
    """列印標題"""
    print("\n" + "=" * 80)
    print("🚀 BERT4Rec 快速訓練工具")
    print("=" * 80)


def check_data() -> dict:
    """檢查資料庫狀態"""
    print("\n📊 檢查資料...")

    if not Path("bert.db").exists():
        print("