# 混合推薦系統整合指南

## 📋 概述

已成功整合 **混合推薦引擎**（Hybrid Recommendation Engine），結合：
- **內容推薦**（Content-Based）：基於 genre/tags 的餘弦相似度
- **BERT 推薦**（可選）：基於 AnimeRecBERT 的序列推薦模型

## ✅ 已完成的工作

### 1. 新增的檔案

- `Lunaris/bert_recommender.py` - BERT 模型包裝器
- `Lunaris/hybrid_recommendation_engine.py` - 混合推薦引擎
- `Lunaris/bert_config.py` - BERT 配置檔案

### 2. 修改的檔案

- `Lunaris/main.py` - 整合混合推薦引擎到 `/recommend` endpoint

### 3. 當前狀態

- ✅ **前端**：`Solaris/src/pages/Recommend.tsx` 已準備好，無需修改
- ✅ **後端**：混合推薦引擎已整合，目前運行在 **Content-Only 模式**
- ⏳ **BERT 模型**：尚未下載（可選功能）

## 🚀 立即使用（Content-Only 模式）

目前系統已經可以正常運作，使用增強版的內容推薦：

```bash
# 1. 啟動後端
cd Lunaris
python main.py

# 2. 啟動前端
cd Solaris
npm run dev
```

**測試流程：**
1. 前端訪問 `http://localhost:5173/recommend`
2. 輸入 AniList 使用者名稱（例如：`senba1000m3`）
3. 選擇年份和季度
4. 點擊「取得推薦」
5. 查看個人化推薦結果和匹配理由

## 🔧 啟用 BERT 推薦（可選）

### 方案 A：不使用 BERT（推薦，已完成）

當前模式已經足夠好用，使用基於內容的推薦：
- ✅ 快速、輕量
- ✅ 可處理新番（不在訓練集中的動畫）
- ✅ 推薦理由清晰易懂
- ✅ 無需下載大型模型檔案

### 方案 B：啟用 BERT（進階功能）

如果想要啟用 BERT 增強推薦：

#### 1. 下載 AnimeRecBERT 模型

```bash
# 創建模型目錄
mkdir -p Lunaris/data/bert_model

# 下載模型（需要 Kaggle API）
# 方法 1: 使用 Kaggle CLI
kaggle datasets download -d ramazanturann/animeratings-mini-54m
unzip animeratings-mini-54m.zip -d Lunaris/data/bert_model/

# 方法 2: 手動下載
# 訪問: https://www.kaggle.com/datasets/ramazanturann/animeratings-mini-54m
# 下載並解壓到 Lunaris/data/bert_model/
```

#### 2. 建立 ID 映射檔案

需要建立 AniList ID 與 Dataset ID 的映射：

```json
// Lunaris/data/bert_model/id_mapping.json
{
  "21": 1,
  "1535": 2,
  // ... AniList ID -> Dataset ID 映射
}
```

**注意**：這需要額外的工作來建立映射關係。

#### 3. 修改配置啟用 BERT

```python
# 在 Lunaris/main.py 中修改第 84 行：
hybrid_rec_engine = HybridRecommendationEngine(
    bert_model_path="data/bert_model/pretrained_bert.pth",
    bert_dataset_path="data/bert_model/dataset.pkl",
    bert_metadata_path="data/bert_model/animes.json",
    use_bert=True  # 改為 True
)
```

#### 4. 重啟後端

```bash
cd Lunaris
python main.py
```

## 📊 API 端點

### 1. 推薦端點（已更新）

```http
POST /recommend
Content-Type: application/json

{
  "username": "senba1000m3",  // 可選，AniList 使用者名稱
  "season": "WINTER",          // 可選，季度
  "year": 2025                 // 可選，年份
}
```

**回應格式：**
```json
{
  "season": "WINTER",
  "year": 2025,
  "display_season": "冬-1 月",
  "recommendations": [
    {
      "id": 123,
      "title": {...},
      "genres": ["Action", "Fantasy"],
      "match_score": 85.5,
      "content_score": 82.0,
      "bert_score": null,  // Content-Only 模式為 null
      "match_reasons": {
        "matched_genres": [
          {"genre": "Action", "weight": 0.85},
          {"genre": "Fantasy", "weight": 0.72}
        ],
        "total_weight": 1.57,
        "top_reason": "你喜歡 Action 和 Fantasy 類型"
      }
    }
  ]
}
```

### 2. 推薦系統狀態端點（新增）

```http
GET /recommend/status
```

**回應範例（Content-Only）：**
```json
{
  "hybrid_engine_available": true,
  "mode": "content_only",
  "bert_enabled": false,
  "bert_weight": 0.0,
  "content_weight": 1.0
}
```

**回應範例（Hybrid）：**
```json
{
  "hybrid_engine_available": true,
  "mode": "hybrid",
  "bert_enabled": true,
  "bert_available": true,
  "bert_weight": 0.6,
  "content_weight": 0.4
}
```

## 🎯 系統架構

### Content-Only 模式（當前）

```
User Input (AniList Username)
    ↓
Fetch User's Anime List (AniList API)
    ↓
Build User Profile (Genre weights from ratings)
    ↓
Fetch Seasonal Anime (AniList API)
    ↓
Calculate Content Similarity (Cosine)
    ↓
Sort & Return Recommendations
```

### Hybrid 模式（啟用 BERT 後）

```
User Input (AniList Username)
    ↓
Fetch User's Anime List (AniList API)
    ↓
┌─────────────────────────┬───────────────────────┐
│  Content