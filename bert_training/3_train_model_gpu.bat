@echo off
chcp 65001 > nul
echo ========================================
echo 步驟 3: 訓練 BERT4Rec 模型 (GPU 版本)
echo ========================================
echo.
echo 此腳本會使用 GPU 加速訓練 BERT4Rec 推薦模型
echo 訓練參數: 200 epochs, batch_size=128, GPU enabled
echo.
echo ----------------------------------------
echo.

python train_model.py --epochs 200 --batch-size 128 --lr 0.001 --gpu

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
echo 注意: 如果沒有 NVIDIA GPU，請使用 3_train_model.bat
echo.
pause
