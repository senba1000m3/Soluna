# BERT4Rec 訓練與使用指南

## 📋 完整流程

### 1. 準備動畫資料（只需執行一次）
```bash
prepare_anime.bat
```
- 從 AniList 抓取 3000 部熱門動畫
- 建立 `bert.db` 資料庫
- 時間：約 10-20 分鐘

---

### 2. 載入使用者資料
```bash
load_users.bat
```
- 從 `datas_user.txt` 讀取使用者名稱
- 抓取每個使用者的動畫列表
- 只保留至少有 30 部動畫的使用者
- 時間：約 2-3 分鐘/使用者

**datas_user.txt 格式範例：**
```
John
Alex
Shadow
mike
Weeb
SenPai
...
```

---

### 3. 訓練模型
```bash
train.bat
```
- 使用載入的資料訓練 BERT4Rec 模型
- 預設參數：20 epochs, batch size 64
- 時間：10-30 分鐘（依資料量和電腦效能）

---

## 📁 訓練完成後的檔案

訓練完成後會在 `bert_models/` 資料夾產生：

```
bert_models/
├── best_model.pth          # 訓練好的模型（約 30-50 MB）
├── item_mappings.pkl       # 動畫 ID 映射檔案（約 40-100 KB）
└── checkpoint_epoch_X.pth  # 訓練過程的檢查點
```

---

## 🚀 如何使用訓練好的模型

### 方法 1: 在 main.py 中使用

```python
from bert_recommender_optimized import OptimizedBERTRecommender
from sqlmodel import Session
from database import engine

# 初始化推薦器
with Session(engine) as session:
    bert_recommender = OptimizedBERTRecommender(
        model_path="bert_models/best_model.pth",
        dataset_path="bert_models/item_mappings.pkl",
        db_session=session,
        device="auto",  # 自動選擇 CPU/GPU
    )
    
    # 取得推薦
    user_anime_ids = [16498, 1535, 101922]  # 使用者看過的動畫 ID
    recommendations = bert_recommender.get_recommendations(
        user_anime_ids=user_anime_ids,
        username="test_user",
        top_k=10,
        use_anilist_ids=True,
    )
    
    # recommendations 格式：
    # [
    #   {"anime_id": 123, "score": 0.95, "title": "..."},
    #   {"anime_id": 456, "score": 0.88, "title": "..."},
    #   ...
    # ]
```

### 方法 2: 整合到現有的推薦引擎

編輯 `hybrid_recommendation_engine.py`：

```python
class HybridRecommendationEngine:
    def __init__(self, ...):
        # ... 現有的初始化代碼 ...
        
        # 加入 BERT 推薦器
        self.bert_recommender = OptimizedBERTRecommender(
            model_path="bert_models/best_model.pth",
            dataset_path="bert_models/item_mappings.pkl",
            db_session=self.session,
        )
    
    async def get_hybrid_recommendations(self, user_anime_ids, ...):
        # 取得 BERT 推薦
        bert_recs = self.bert_recommender.get_recommendations(
            user_anime_ids=user_anime_ids,
            top_k=50,
            use_anilist_ids=True,
        )
        
        # 合併其他推薦來源
        # ... 你的混合邏輯 ...
```

---

## 🔄 重新訓練

當有新的使用者資料時：

```bash
# 1. 更新 datas_user.txt（加入新使用者）
# 2. 載入新使用者
load_users.bat

# 3. 重新訓練
train.bat
```

**建議重訓頻率：**
- 每週一次：如果有持續新增使用者
- 每月一次：如果使用者變化不大

---

## 📊 檢查訓練資料

```bash
uv run python -c "from sqlmodel import Session, select, create_engine; from prepare_bert_dataset import BERTAnime, BERTUserAnimeList; engine = create_engine('sqlite:///bert.db'); session = Session(engine); anime_count = len(session.exec(select(BERTAnime)).all()); user_ids = session.exec(select(BERTUserAnimeList.user_id).distinct()).all(); user_count = len(user_ids); record_count = len(session.exec(select(BERTUserAnimeList)).all()); print(f'動畫數量: {anime_count}'); print(f'使用者數量: {user_count}'); print(f'訓練記錄: {record_count}'); print(f'平均每使用者: {record_count/user_count:.1f} 部' if user_count > 0 else '')"
```

---

## ⚠️ 注意事項

1. **最低資料需求：**
   - 動畫：至少 1000 部（建議 3000 部）
   - 使用者：至少 30 人（建議 50+ 人）
   - 每使用者：至少 30 部動畫

2. **訓練時間：**
   - 30 使用者 + 20 epochs ≈ 10-15 分鐘
   - 50 使用者 + 20 epochs ≈ 20-30 分鐘

3. **模型檔案：**
   - `best_model.pth` 必須存在才能使用推薦功能
   - 不要手動修改 `item_mappings.pkl`

---

## 🐛 常見問題

### Q: 訓練很慢，每個 epoch 超過 30 秒
**A:** 這是正常的，因為使用 CPU 訓練。如果要加速：
- 減少 epochs 數量
- 或等待 GPU 版本

### Q: 訓練失敗，顯示 "Target X is out of bounds"
**A:** 資料有問題，重新載入使用者：
```bash
load_users.bat
```

### Q: 模型推薦結果都一樣
**A:** 可能原因：
- 訓練資料太少（增加使用者）
- 訓練不充分（增加 epochs）
- 使用者太相似（增加多樣性的使用者）

### Q: 找不到 bert_models/best_model.pth
**A:** 訓練尚未完成或失敗，檢查：
1. 是否執行過 `train.bat`
2. 訓練是否成功完成
3. `train_bert_model.log` 中的錯誤訊息

---

## 📞 需要幫助？

檢查日誌檔案：
- `prepare_bert_dataset.log` - 動畫資料準備
- `load_users.log` - 使用者資料載入
- `train_bert_model.log` - 模型訓練

---

## 🎉 下一步

訓練完成後：
1. 整合到你的 `main.py` 推薦 API
2. 測試推薦品質
3. 定期重新訓練以包含新資料