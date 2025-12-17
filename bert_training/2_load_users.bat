@echo off
chcp 65001 > nul
echo ========================================
echo 步驟 2: 載入用戶數據
echo ========================================
echo.
echo 此腳本會從 datas_user.txt 讀取用戶名單
echo 並抓取每個用戶的動畫觀看記錄
echo.
echo ----------------------------------------
echo.

python load_users.py --min-anime 20

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
echo 下一步: 執行 3_train_model.bat 開始訓練模型
echo.
pause
