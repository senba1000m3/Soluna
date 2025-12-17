@echo off
chcp 65001 > nul
echo ========================================
echo 步驟 3: 訓練 BERT4Rec 模型
echo ========================================
echo.
echo 此腳本會開始訓練 BERT4Rec 推薦模型
echo 訓練參數: 200 epochs, batch_size=64
echo.
echo ----------------------------------------
echo.

python train_model.py --epochs 200 --batch-size 64 --lr 0.001

echo.
echo ========================================
echo 訓練完成！
echo ========================================
echo.
echo 輸出文件位置:
echo   - 模型: output/models/
echo   - 圖表: output/plots/
echo   - 日誌: training.log
echo.
echo 查看訓練結果:
echo   1. 檢查 output/plots/combined_metrics.png
echo   2. 檢查 output/plots/training_metrics.json
echo   3. 檢查 training.log
echo.
pause
