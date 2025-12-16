# BERT 推薦系統運作流程與架構文檔

## 📋 概述

Soluna 使用混合推薦引擎，結合了 **BERT4Rec 序列推薦模型**和**內容特徵推薦**，為使用者提供個性化的新番推薦。

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Recommendation Engine              │
│                                                               │
│  ┌────────────────────┐        ┌─────────────────────────┐  │
│  │  BERT Recommender  │        │  Content Recommender    │  │
│  │                    │        │                         │  │
│  │  • BERT4Rec 模型   │        │  • Genre 分析           │  │
│  │  • 序列預測        │        │  • Tag 分析             │  │
│  │  • ID 映射         │        │  • Studio 分析          │  │
│  └────────────────────┘        └─────────────────────────┘  │
│           ↓                              ↓                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Feature Fusion (特徵融合)                   │    │
│  │     BERT (60%) + Content (40%) = Final Score       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 完整推薦流程

### 階段 1: 初始化與資料載入

```python
hybrid_engine = HybridRecommendationEngine(
    bert_model_path="path/to/bert_model.pth",      # BERT4Rec 預訓練模型
    bert_dataset_path="path/to/dataset.pkl",       # ID 映射資料集
    bert_metadata_path="path/to/animes.json",      # 動畫 metadata
    use_bert=True                                   # 是否啟用 BERT
)
```

**載入的資料**:
1. **BERT 模型**: PyTorch 預訓練的 BERT4Rec 模型
2. **資料集**: 包含 `smap` (item_id → sequential_id) 的映射
3. **Metadata**: 動畫的類型、標籤、製作公司等資訊

---

### 階段 2: 建立使用者 Profile

#### 2.1 內容 Profile (Content-Based)

**目的**: 分析使用者過去觀看的動畫，找出偏好特徵

**輸入**: 使用者的 AniList 動畫列表
```python
user_list = [
    {
        "media": {
            "id": 1,
            "title": "Cowboy Bebop",
            "genres": ["Action", "Sci-Fi"],
            "tags": [{"name": "Space"}, {"name": "Bounty Hunters"}],
            "studios": [{"name": "Sunrise"}]
        },
        "score": 9,
        "status": "COMPLETED"
    },
    # ... 更多動畫
]
```

**處理流程**:
```
1. 過濾高評分動畫 (score >= 7)
   └─→ 獲取「喜歡的作品」

2. 提取特徵
   ├─→ Genres: ["Action", "Sci-Fi", ...]
   ├─→ Tags: ["Space", "Mecha", ...]
   └─→ Studios: ["Sunrise", "Bones", ...]

3. 計算特徵權重
   └─→ TF-IDF 或計數加權
       例如: Genre_Action: 0.35, Genre_Sci-Fi: 0.28

4. 輸出 Content Profile
   {
     "Genre_Action": 0.35,
     "Genre_Sci-Fi": 0.28,
     "Tag_Space": 0.22,
     "Studio_Sunrise": 0.15,
     ...
   }
```

#### 2.2 BERT Profile (Sequence-Based)

**目的**: 使用 BERT 模型理解使用者的觀看序列模式，發現隱藏偏好

**輸入**: 使用者觀看過的動畫 ID 列表
```python
user_anime_ids = [1, 5, 6, 15, 16, ...]  # AniList IDs
```

**BERT 推薦流程**:

```
┌────────────────────────────────────────────────────────────┐
│  階段 1: ID 映射                                            │
├────────────────────────────────────────────────────────────┤
│  AniList ID → Dataset ID                                   │
│  [1, 5, 6, ...] → [101, 205, 306, ...]                    │
│                                                             │
│  ✓ 成功映射 45/50 個 ID                                     │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│  階段 2: 準備 BERT 輸入序列                                 │
├────────────────────────────────────────────────────────────┤
│  input_seq = [CLS] item1 item2 ... itemN [MASK]           │
│                                                             │
│  • 截斷/填充到固定長度 (max_len=200)                       │
│  • 轉換為 PyTorch Tensor                                   │
│  • 移動到計算設備 (CPU/GPU)                                │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│  階段 3: BERT4Rec 模型推理                                  │
├────────────────────────────────────────────────────────────┤
│  Model Architecture:                                        │
│  ┌──────────────────────────────────────────────┐         │
│  │  Embedding Layer                              │         │
│  │    ↓                                          │         │
│  │  Multi-Head Self-Attention (x N layers)      │         │
│  │    ↓                                          │         │
│  │  Feed-Forward Network                         │         │
│  │    ↓                                          │         │
│  │  Output Layer → Logits for all items         │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│  Input:  [101, 205, 306, ...]                             │
│  Output: [0.23, 0.45, 0.89, ...] (scores for all items)  │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│  階段 4: 取 Top-K 推薦 (K=50)                               │
├────────────────────────────────────────────────────────────┤
│  scores.argsort()[-50:][::-1]                              │
│  → [3456, 1234, 7890, ...]  (Dataset IDs)                 │
│                                                             │
│  映射回 AniList ID:                                         │
│  [3456, 1234, 7890, ...] → [23, 45, 67, ...]             │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│  階段 5: 提取參考動畫特徵                                   │
├────────────────────────────────────────────────────────────┤
│  從 Top-50 推薦中提取特徵:                                  │
│  ├─ Genres: {"Action": 35, "Sci-Fi": 28, ...}            │
│  ├─ Tags: {"Mecha": 20, "Space": 15, ...}                │
│  └─ Studios: {"Sunrise": 12, "Bones": 8, ...}            │
│                                                             │
│  根據推薦分數加權:                                          │
│  Genre_Action: (35 × avg_score) / total_score = 0.42      │
└────────────────────────────────────────────────────────────┘
```

**BERT Profile 輸出**:
```python
{
    "genres": {
        "Action": 0.42,
        "Sci-Fi": 0.35,
        "Mecha": 0.28,
        ...
    },
    "tags": {
        "Space": 0.30,
        "Military": 0.25,
        ...
    },
    "studios": {
        "Sunrise": 0.38,
        "Bones": 0.22,
        ...
    }
}
```

---

### 階段 3: 評分新番動畫

**輸入**: 當季新番列表

```python
seasonal_anime = [
    {
        "id": 150462,
        "title": {"romaji": "Sousou no Frieren"},
        "genres": ["Adventure", "Fantasy"],
        "tags": [{"name": "Magic"}, {"name": "Demons"}],
        "studios": [{"name": "Madhouse"}]
    },
    # ... 更多新番
]
```

#### 3.1 內容分數計算 (Content Score)

**方法**: 計算新番特徵與內容 Profile 的相似度

```python
def calculate_content_score(anime, content_profile):
    # 1. 提取新番特徵向量
    anime_features = extract_features(anime)
    # anime_features = {
    #     "Genre_Adventure": 1.0,
    #     "Genre_Fantasy": 1.0,
    #     "Tag_Magic": 1.0,
    #     "Studio_Madhouse": 1.0
    # }
    
    # 2. 計算餘弦相似度
    similarity = cosine_similarity(anime_features, content_profile)
    
    # 3. 正規化到 0-100
    content_score = similarity * 100
    
    return content_score
```

**計算過程**:
```
新番: Sousou no Frieren
  Genres: [Adventure, Fantasy]
  Tags: [Magic, Demons]
  Studio: Madhouse

使用者 Content Profile:
  Genre_Adventure: 0.28  ✓ 匹配！
  Genre_Fantasy: 0.35    ✓ 匹配！
  Tag_Magic: 0.20        ✓ 匹配！
  Studio_Madhouse: 0.15  ✓ 匹配！

相似度計算:
  dot_product = (1.0 × 0.28) + (1.0 × 0.35) + (1.0 × 0.20) + (1.0 × 0.15)
              = 0.98
  
  cosine_similarity = dot_product / (||anime|| × ||profile||)
                    ≈ 0.75
  
  content_score = 0.75 × 100 = 75
```

#### 3.2 BERT 分數計算 (BERT Score)

**方法**: 計算新番特徵與 BERT Profile 的相似度

```python
def calculate_bert_score(anime, bert_profile):
    # 1. 提取新番特徵
    anime_genres = set(anime.get("genres", []))
    anime_tags = set([t["name"] for t in anime.get("tags", [])])
    
    # 2. 計算與 BERT Profile 的重疊度
    genre_score = sum(bert_profile["genres"].get(g, 0) for g in anime_genres)
    tag_score = sum(bert_profile["tags"].get(t, 0) for t in anime_tags)
    
    # 3. 加權平均
    bert_score = (genre_score * 0.6 + tag_score * 0.4) * 100
    
    return bert_score
```

**計算過程**:
```
新番: Sousou no Frieren
  Genres: [Adventure, Fantasy]
  Tags: [Magic, Demons]

BERT Profile:
  genres: {Adventure: 0.32, Fantasy: 0.40, ...}
  tags: {Magic: 0.28, Demons: 0.15, ...}

計算:
  genre_score = 0.32 + 0.40 = 0.72
  tag_score = 0.28 + 0.15 = 0.43
  
  bert_score = (0.72 × 0.6 + 0.43 × 0.4) × 100
             = (0.432 + 0.172) × 100
             = 60.4
```

#### 3.3 最終分數融合

```python
# 混合權重
BERT_WEIGHT = 0.6      # BERT 佔 60%
CONTENT_WEIGHT = 0.4   # 內容佔 40%

final_score = content_score × CONTENT_WEIGHT + bert_score × BERT_WEIGHT

# 範例計算
final_score = 75 × 0.4 + 60.4 × 0.6
            = 30 + 36.24
            = 66.24
```

**為什麼 BERT 權重較高？**
- BERT 能捕捉**序列模式**和**隱藏偏好**
- 內容推薦只看表面特徵匹配
- BERT 推薦的動畫通常「意想不到但很適合」

---

### 階段 4: 生成推薦理由

```python
def generate_match_reasons(anime, content_profile, bert_profile):
    reasons = []
    
    # 檢查 Genre 匹配
    for genre in anime["genres"]:
        if f"Genre_{genre}" in content_profile:
            weight = content_profile[f"Genre_{genre}"]
            if weight > 0.2:
                reasons.append(f"你喜歡 {genre} 類型")
    
    # 檢查 BERT 推薦的特殊標籤
    if bert_profile:
        for tag in anime.get("tags", []):
            tag_name = tag["name"]
            if tag_name in bert_profile.get("tags", {}):
                reasons.append(f"BERT 模型認為你會喜歡 {tag_name}")
    
    return reasons
```

**輸出範例**:
```python
{
    "id": 150462,
    "title": {"romaji": "Sousou no Frieren"},
    "match_score": 66.24,
    "content_score": 75.0,
    "bert_score": 60.4,
    "match_reasons": [
        "你喜歡 Adventure 類型",
        "你喜歡 Fantasy 類型",
        "BERT 模型認為你會喜歡 Magic",
        "製作公司 Madhouse 的作品你通常很喜歡"
    ]
}
```

---

### 階段 5: 排序與返回

```python
# 按 match_score 降序排序
scored_anime.sort(key=lambda x: x["match_score"], reverse=True)

# 返回 Top-N
return scored_anime[:top_n]
```

---

## 🎯 BERT4Rec 模型詳解

### 模型架構

```
Input Sequence: [item₁, item₂, item₃, ..., itemₙ]
       ↓
┌──────────────────────────────────────────┐
│  Embedding Layer                          │
│  每個 item → dense vector (dim=256)      │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Positional Encoding                      │
│  添加位置資訊                             │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Transformer Encoder (N layers)          │
│  ┌────────────────────────────────────┐  │
│  │  Multi-Head Self-Attention         │  │
│  │    ↓                               │  │
│  │  Add & Normalize                   │  │
│  │    ↓                               │  │
│  │  Feed-Forward Network              │  │
│  │    ↓                               │  │
│  │  Add & Normalize                   │  │
│  └────────────────────────────────────┘  │
│  (重複 N 次)                              │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Output Layer                             │
│  → Logits for all items in catalog       │
└──────────────────────────────────────────┘
```

### Self-Attention 機制

**為什麼 BERT 能理解序列模式？**

```
使用者觀看序列: [Attack on Titan, Death Note, Code Geass, Steins;Gate]

Self-Attention 會學習:
  • Attack on Titan 和 Death Note 都是「黑暗」、「懸疑」類型
  • Death Note 和 Code Geass 都有「智鬥」元素
  • Code Geass 和 Steins;Gate 都是「Sci-Fi」

因此推薦: 
  → Psycho-Pass (黑暗 + 懸疑 + Sci-Fi)
  → Monster (黑暗 + 懸疑 + 智鬥)
  → Ergo Proxy (黑暗 + Sci-Fi)
```

---

## ⚡ 性能考量與優化建議

### 當前瓶頸

| 階段 | 時間 | 瓶頸 |
|------|------|------|
| ID 映射 | ~2s | 需要查詢映射表 |
| BERT 推理 | ~5-10s | 模型計算密集 |
| 特徵提取 | ~3s | 需要載入 metadata |
| 內容評分 | ~1s | 餘弦相似度計算 |
| **總計** | **~11-16s** | |

### 優化策略

#### 1. 模型加速

```python
# ❌ 當前: CPU 推理
model = torch.load(model_path, map_location='cpu')

# ✅ 優化: GPU 推理 (5-10x 加速)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.half()  # FP16 推理 (2x 加速)

# ✅ 進階: ONNX 推理 (3-5x 加速)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

**效果**: 推理時間從 10s → 1-2s

#### 2. 批次處理

```python
# ❌ 當前: 逐一評分
for anime in seasonal_anime:
    content_score = calculate_content_score(anime, profile)
    bert_score = calculate_bert_score(anime, bert_profile)

# ✅ 優化: 批次計算
anime_features = extract_features_batch(seasonal_anime)  # 一次提取所有特徵
content_scores = cosine_similarity_batch(anime_features, profile)  # 向量化計算
bert_scores = calculate_bert_scores_batch(anime_features, bert_profile)
```

**效果**: 評分時間從 3s → 0.5s

#### 3. 快取機制

```python
# ✅ 快取 BERT 推薦結果
@lru_cache(maxsize=1000)
def get_bert_recommendations_cached(user_id: int, top_k: int):
    return bert_recommender.get_recommendations(...)

# ✅ 快取特徵提取
feature_cache = {}
def get_anime_features_cached(anime_id):
    if anime_id not in feature_cache:
        feature_cache[anime_id] = extract_features(anime_id)
    return feature_cache[anime_id]
```

**效果**: 重複查詢時間從 15s → 0.1s

#### 4. 預計算 Profile

```python
# ❌ 當前: 每次請求都重新計算 profile
profile = build_user_profile(user_list)

# ✅ 優化: 預計算並存入 DB
def update_user_profile(user_id):
    """使用者列表更新時觸發"""
    profile = build_user_profile(user_list)
    db.save_profile(user_id, profile)

def get_recommendations(user_id):
    """直接從 DB 讀取預計算的 profile"""
    profile = db.load_profile(user_id)
    return score_anime(profile, seasonal_anime)
```

**效果**: Profile 建立時間從 5s → 0s (預計算)

#### 5. 增量更新

```python
# ✅ 只在必要時重新計算
def should_update_profile(user_id):
    last_update = db.get_last_profile_update(user_id)
    last_list_update = anilist.get_last_activity(user_id)
    return last_list_update > last_update

if should_update_profile(user_id):
    update_user_profile(user_id)
```

#### 6. 輕量級模型

```python
# 選項 1: 使用更小的 BERT 模型
# BERT-Base (110M 參數) → BERT-Tiny (4.4M 參數)

# 選項 2: 知識蒸餾
# 訓練一個小模型模仿大模型的行為

# 選項 3: 量化
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**效果**: 模型大小 440MB → 110MB，推理速度提升 2-4x

---

## 📊 品質改進建議

### 1. 更細緻的特徵工程

```python
# ❌ 當前: 只考慮 presence (有/無)
anime_features = {
    "Genre_Action": 1.0,
    "Genre_Fantasy": 1.0
}

# ✅ 改進: 考慮使用者評分
anime_features = {
    "Genre_Action": user_score / 10.0,  # 0.9 for score=9
    "Genre_Fantasy": user_score / 10.0
}

# ✅✅ 進階: 考慮觀看完成度
anime_features = {
    "Genre_Action": (user_score / 10.0) * (progress / episodes),
    "Genre_Fantasy": (user_score / 10.0) * (progress / episodes)
}
```

### 2. 時間衰減

```python
# ✅ 考慮觀看時間，近期的動畫權重更高
import datetime

def calculate_time_weight(completed_at):
    days_ago = (datetime.now() - completed_at).days
    decay_factor = 0.95  # 每天衰減 5%
    return decay_factor ** days_ago

feature_weight = base_weight * calculate_time_weight(completed_at)
```

### 3. 負向過濾

```python
# ✅ 學習使用者「不喜歡」的特徵
low_scored_anime = [a for a in user_list if a["score"] < 5]
negative_profile = build_profile(low_scored_anime)

# 降低匹配負向特徵的動畫分數
if has_negative_features(anime, negative_profile):
    final_score *= 0.7  # 降低 30%
```

### 4. 多樣性控制

```python
# ✅ 避免推薦過於相似的動畫
def diversify_recommendations(recommendations, diversity_factor=0.3):
    """
    在保持高分的同時增加多樣性
    """
    diversified = []
    seen_genres = set()
    
    for rec in recommendations:
        genres = set(rec["genres"])
        
        # 如果類型重複太多，降低分數
        overlap = len(genres & seen_genres) / len(genres)
        diversity_penalty = 1.0 - (overlap * diversity_factor)
        
        rec["match_score"] *= diversity_penalty
        seen_genres.update(genres)
        diversified.append(rec)
    
    return sorted(diversified, key=lambda x: x["match_score"], reverse=True)
```

### 5. A/B 測試框架

```python
# ✅ 建立實驗框架
class RecommenderExperiment:
    def __init__(self, variant="control"):
        self.variant = variant
    
    def get_weights(self):
        if self.variant == "control":
            return {"bert": 0.6, "content": 0.4}
        elif self.variant == "bert_heavy":
            return {"bert": 0.8, "content": 0.2}
        elif self.variant == "balanced":
            return {"bert": 0.5, "content": 0.5}
    
    def log_recommendation(self, user_id, anime_id, score, clicked):
        """記錄使用者是否點擊了推薦"""
        db.save_experiment_result(
            variant=self.variant,
            user_id=user_id,
            anime_id=anime_id,
            score=score,
            clicked=clicked
        )
```

### 6. 冷啟動處理

```python
def recommend_with_cold_start(user_list, seasonal_anime):
    # 如果使用者資料太少
    if len(user_list) < 10:
        # 使用人氣推薦 + 少量個性化
        popular_anime = get_popular_seasonal_anime(seasonal_anime)
        
        if len(user_list) > 0:
            # 混合一些個性化推薦
            personal_recs = hybrid_recommend(user_list, seasonal_anime)
            return merge_recommendations(
                popular_anime, 
                personal_recs, 
                popular_weight=0.7
            )
        else:
            return popular_anime
    
    # 正常推薦流程
    return hybrid_recommend(user_list, seasonal_anime)
```

---

## 🧪 評估指標

### 離線評估

```python
# 1. 準確率 (Precision@K)
def precision_at_k(predicted, actual, k=10):
    """前 K 個推薦中有多少是使用者實際喜歡的"""
    top_k = predicted[:k]
    hits = len(set(top_k) & set(actual))
    return hits / k

# 2. 召回率 (Recall@K)
def recall_at_k(predicted, actual, k=10):
    """使用者喜歡的動畫中有多少被推薦了"""
    top_k = predicted[:k]
    hits = len(set(top_k) & set(actual))
    return hits / len(actual)

# 3. NDCG (Normalized Discounted Cumulative Gain)
def ndcg_at_k(predicted, actual_scores, k=10):
    """考慮排序位置的評估指標"""
    dcg = sum(
        actual_scores.get(pred, 0) / np.log2(i + 2)
        for i, pred in enumerate(predicted[:k])
    )
    ideal_dcg = sum(
        score / np.log2(i + 2)
        for i, score in enumerate(sorted(actual_scores.values(), reverse=True)[:k])
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0
```

### 線上評估

```python
# 4. 點擊率 (CTR)
CTR = (點擊推薦動畫的次數) / (推薦展示的次數)

# 5. 轉換率
Conversion_Rate = (加入列表的動畫數) / (點擊的推薦數)

# 6. 使用者滿意度
# 定期問卷調查推薦品質
```

---

## 🔧 部署配置

### 開發環境

```python
# config/development.py
BERT_CONFIG = {
    "use_bert": False,  # 開發時關閉 BERT 加速測試
    "model_path": None,
    "dataset_path": None,
    "device": "cpu"
}
```

### 生產環境

```python
# config/production.py
BERT_CONFIG = {
    "use_bert": True,
    "model_path": "/models/bert4rec_anime.pth",
    "dataset_path": "/data/anime_dataset.pkl",
    "metadata_path": "/data/anime_metadata.json",
    "device": "cuda",  # 使用 GPU
    "batch_size": 32,
    "cache_ttl": 3600,  # 快取 1 小時
}
```

---

## 📚 參考資料

- **BERT4Rec 論文**: [BERT4Rec: Sequential Recommendation with BERT](https://arxiv.org/abs/1904.06690)
- **Transformer 架構**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- **推薦系統綜述**: [Deep Learning based Recommender System: A Survey](https://arxiv.org/abs/1707.07435)

---

## ✅ 總結

### 系統優勢

✅ **雙引擎推薦**: BERT 序列模型 + 內容特徵，互補優勢  
✅ **可解釋性**: 生成推薦理由，使用者知道為什麼被推薦  
✅ **彈性配置**: 可動態調整 BERT/內容權重  
✅ **Fallback 機制**: BERT 不可用時自動降級到內容推薦  

### 改進方向

🔄 **性能優化**: GPU 推理、批次處理、快取機制  
🔄 **品質提升**: 時間衰減、負向過濾、多樣性控制  
🔄 **冷啟動**: 為新使用者提供更好的初始推薦  
🔄 **A/B 測試**: 持續實驗優化權重和參數  

---

**文檔版本**: v1.0  
**最後更新**: 2024
**維護者**: Soluna 開發團隊