@echo off
chcp 65001 > nul
echo ================================================================================
echo BERT4Rec 訓練環境設置
echo ================================================================================
echo.
echo 此腳本會幫助您設置訓練環境
echo.
echo ================================================================================
echo.

echo 檢查 Python 安裝...
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 找不到 Python！請先安裝 Python 3.8 或更高版本
    echo    下載地址: https://www.python.org/downloads/
    goto :error
)

python --version
echo ✅ Python 已安裝
echo.

echo ================================================================================
echo 安裝依賴套件
echo ================================================================================
echo.
echo 正在安裝所需套件... 這可能需要幾分鐘
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ❌ 套件安裝失敗！
    goto :error
)

echo.
echo ✅ 所有套件安裝完成
echo.

echo ================================================================================
echo 檢查 PyTorch GPU 支援
echo ================================================================================
echo.

python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU 數量:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU 名稱:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo ⚠️  無法檢查 GPU 狀態
) else (
    echo.
    echo 如果顯示 "CUDA 可用: True"，您可以使用 GPU 加速訓練
    echo 使用 3_train_model_gpu.bat 啟動 GPU 訓練
)

echo.

echo ================================================================================
echo 創建必要目錄
echo ================================================================================
echo.

if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "output\models" mkdir output\models
if not exist "output\plots" mkdir output\plots
if not exist "output\logs" mkdir output\logs
if not exist "output\checkpoints" mkdir output\checkpoints

echo ✅ 目錄結構已創建:
echo    - data/           (資料庫存放位置)
echo    - output/models/  (訓練模型輸出)
echo    - output/plots/   (訓練圖表輸出)
echo    - output/logs/    (日誌輸出)
echo    - output/checkpoints/ (訓練檢查點)
echo.

echo ================================================================================
echo 檢查必要文件
echo ================================================================================
echo.

if not exist "datas_user.txt" (
    echo ⚠️  找不到 datas_user.txt
    echo    請確保該文件存在且包含用戶名單（每行一個用戶名）
    echo.
) else (
    echo ✅ datas_user.txt 已存在
    for /f %%i in ('type datas_user.txt ^| find /c /v ""') do set user_count=%%i
    echo    包含 %user_count% 個用戶
    echo.
)

echo ================================================================================
echo 🎉 環境設置完成！
echo ================================================================================
echo.
echo 接下來的步驟：
echo.
echo 📝 方式 1: 分步執行（推薦用於首次使用）
echo    1. 執行 1_prepare_anime.bat   - 準備動畫數據
echo    2. 執行 2_load_users.bat       - 載入用戶數據
echo    3. 執行 3_train_model.bat      - 訓練模型（CPU）
echo       或 3_train_model_gpu.bat    - 訓練模型（GPU）
echo.
echo 🚀 方式 2: 一鍵執行（完整自動化流程）
echo    執行 run_all.bat - 自動完成所有步驟
echo.
echo 📖 詳細說明：
echo    請查看 README.md 了解完整使用說明
echo.
echo ⚙️  配置調整：
echo    修改 config.py 可以調整訓練參數
echo.
goto :end

:error
echo.
echo ================================================================================
echo ❌ 設置過程中發生錯誤
echo ================================================================================
echo.
echo 請檢查並修復錯誤後重新執行此腳本
echo.
goto :end

:end
echo.
pause
