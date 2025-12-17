@echo off
chcp 65001 > nul
echo ========================================
echo 步驟 1: 準備動畫數據
echo ========================================
echo.
echo 此腳本會從 AniList 抓取 3000 部熱門動畫
echo 並儲存到資料庫中供訓練使用
echo.
echo ----------------------------------------
echo.

python prepare_dataset.py --count 3000

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
echo 下一步: 執行 2_load_users.bat 載入用戶數據
echo.
pause
