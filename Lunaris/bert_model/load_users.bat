@echo off
chcp 65001 > nul
echo ================================================================================
echo 👥 載入使用者資料
echo ================================================================================
echo.

REM 檢查 datas_user.txt 是否存在
if not exist "datas_user.txt" (
    echo ❌ 錯誤: datas_user.txt 不存在
    echo.
    echo 請先建立 datas_user.txt 檔案，每行一個使用者名稱
    echo 範例:
    echo   John
    echo   Alex
    echo   Shadow
    echo   ...
    echo.
    pause
    exit /b 1
)

REM 檢查 bert.db 是否存在
if not exist "bert.db" (
    echo ❌ 錯誤: bert.db 不存在
    echo.
    echo 請先執行 prepare_anime.bat 準備動畫資料
    echo.
    pause
    exit /b 1
)

echo 此腳本會從 datas_user.txt 讀取使用者名稱
echo 並從 AniList 抓取他們的動畫列表
echo.
echo 設定:
echo   - 最少動畫數: 30 部
echo   - 每個使用者間隔: 2 秒 (避免 API 限制)
echo.
echo 預估時間: 視使用者數量而定 (約 2-3 分鐘/使用者)
echo.
echo ================================================================================
echo.

REM 執行載入腳本
uv run python load_users_from_file.py --min-anime 30

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo ✅ 使用者資料載入完成！
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
    echo ❌ 載入失敗或部分失敗
    echo ================================================================================
    echo.
    echo 請檢查:
    echo   1. datas_user.txt 中的使用者名稱是否正確
    echo   2. 網路連線是否正常
    echo   3. load_users.log 檔案中的錯誤訊息
    echo.
)

pause
