@echo off
chcp 65001 > nul
echo ================================================================================
echo 🤖 BERT4Rec 模型訓練
echo ================================================================================
echo.

REM 檢查資料庫是否存在
if not exist "bert.db" (
    echo ❌ 錯誤: bert.db 不存在
    echo.
    echo 請先執行以下步驟：
    echo   1. 準備動畫資料: prepare_anime.bat
    echo   2. 載入使用者資料: load_users.bat
    echo.
    pause
    exit /b 1
)

echo 📊 檢查資料庫狀態...
uv run python -c "from sqlmodel import Session, select, create_engine; from prepare_bert_dataset import BERTAnime, BERTUserAnimeList; engine = create_engine('sqlite:///bert.db'); session = Session(engine); anime_count = len(session.exec(select(BERTAnime)).all()); user_ids = session.exec(select(BERTUserAnimeList.user_id).distinct()).all(); user_count = len(user_ids); record_count = len(session.exec(select(BERTUserAnimeList)).all()); print(f'  動畫數量: {anime_count}'); print(f'  使用者數量: {user_count}'); print(f'  訓練記錄: {record_count}'); print(f'  平均每使用者: {record_count/user_count:.1f} 部動畫' if user_count > 0 else '')"
echo.

echo ================================================================================
echo 開始訓練模型...
echo ================================================================================
echo.
echo 訓練參數:
echo   - Epochs: 20
echo   - Batch Size: 8
echo   - Hidden Size: 256
echo   - Attention Heads: 4
echo   - Transformer Layers: 2
echo.
echo 預估時間: 20-40 分鐘（取決於資料量和電腦效能）
echo.
echo ================================================================================
echo.

REM 執行訓練
uv run python train_bert_model.py --epochs 20 --batch-size 8 --hidden-size 256 --num-heads 4 --num-layers 2

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo ✅ 訓練完成！
    echo ================================================================================
    echo.
    echo 模型檔案位置: trained_models\
    echo   - best_model.pth      (最佳模型)
    echo   - item_mappings.pkl   (動畫 ID 映射)
    echo.
    echo 下一步:
    echo   1. 查看訓練日誌: train_bert_model.log
    echo   2. 使用模型進行推薦 (整合到 main.py)
    echo.
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo ❌ 訓練失敗
    echo ================================================================================
    echo.
    echo 請檢查:
    echo   1. train_bert_model.log 檔案
    echo   2. 資料庫是否有足夠的資料
    echo.
)

pause
