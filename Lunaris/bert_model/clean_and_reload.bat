@echo off
chcp 65001 > nul
echo ================================================================================
echo 🧹 清理並重新載入 BERT 資料
echo ================================================================================
echo.
echo ⚠️  警告: 此操作會刪除所有現有的使用者訓練資料
echo.
echo 此腳本會：
echo   1. 保留動畫資料 (BERTAnime)
echo   2. 刪除所有使用者-動畫記錄 (包含 mock 資料)
echo   3. 從 datas_user.txt 重新載入使用者資料
echo.
echo ================================================================================
echo.

REM 檢查 bert.db 是否存在
if not exist "bert.db" (
    echo ❌ 錯誤: bert.db 不存在
    echo.
    echo 請先執行 prepare_anime.bat 準備動畫資料
    echo.
    pause
    exit /b 1
)

REM 檢查 datas_user.txt 是否存在
if not exist "datas_user.txt" (
    echo ❌ 錯誤: datas_user.txt 不存在
    echo.
    echo 請先建立 datas_user.txt 檔案，每行一個使用者名稱
    echo.
    pause
    exit /b 1
)

echo 按任意鍵繼續，或關閉視窗取消...
pause > nul
echo.

REM 執行清理和重新載入
uv run python clean_and_reload.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo ✅ 清理並重新載入完成！
    echo ================================================================================
    echo.
    echo 資料庫: bert.db (已更新)
    echo.
    echo 下一步:
    echo   執行 train.bat 開始訓練模型
    echo.
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo ❌ 執行失敗
    echo ================================================================================
    echo.
    echo 請檢查錯誤訊息
    echo.
)

pause
