# Soluna 前端後端架構文檔

## 前端頁面結構

### 1. Soluna.tsx - 首頁
**功能**: 系統首頁，展示主要功能入口

**使用的組件**:
- 無特定後端 API 調用
- 純展示頁面

---

### 2. QuickIDSettings.tsx - 快速 ID 設定
**功能**: 管理主 ID 和常用 ID

**後端 API**:
- `POST /global-user/login` - 登入主 ID
  - 輸入: `{ anilist_username, anilist_id, avatar }`
  - 返回: `{ user, quickIds[] }`
  
- `GET /global-user/{anilist_id}/quick-ids` - 獲取常用 ID 列表
  - 返回: `QuickID[]`

- `POST /quick-ids` - 新增常用 ID
  - 輸入: `{ owner_anilist_id, anilist_username, anilist_id, avatar, nickname? }`
  - 返回: `QuickID`

- `DELETE /quick-ids/{id}` - 刪除常用 ID
  
- `PATCH /quick-ids/{id}?nickname=xxx` - 更新常用 ID 暱稱
  - 返回: 更新後的 `QuickID`

**外部 API**:
- `https://graphql.anilist.co` - AniList GraphQL API
  - 用於驗證使用者名稱和獲取頭像

**使用的 Context**:
- `AuthContext` (GlobalUserContext)
  - 管理主 ID 和快速 ID 狀態
  - 處理登入登出邏輯

**資料庫表**:
- `GlobalUser` - 主 ID 資料
- `QuickID` - 常用 ID 資料

---

### 3. Recommend.tsx - 動漫推薦
**功能**: 基於使用者觀看歷史推薦動漫

**後端 API**:
- `POST /recommend` - 獲取推薦列表
  - 輸入: `{ username: string }`
  - 返回: `{ username, recommendations[], model_info }`

**後端引擎**:
- `HybridRecommendationEngine`
  - `BERTRecommender` (80%): 序列推薦
  - `CollaborativeRecommendationEngine` (20%): 協同過濾

**資料流程**:
1. 前端發送使用者名稱
2. 後端調用 `anilist_client.get_user_anime_list()` 獲取觀看清單
3. 調用 `fetch_and_store_user_data()` 儲存資料
4. `HybridRecommendationEngine.get_recommendations()` 生成推薦
5. 返回推薦列表給前端

**資料庫表**:
- `User` - 使用者資料
- `UserRating` - 使用者評分記錄
- `Anime` - 動漫資料
- `bert.db` (BERT 模型專用資料庫)

---

### 4. DropPredict.tsx - 棄番預測
**功能**: 預測使用者可能棄番的動漫

**後端 API**:
- `POST /analyze_drops` - 分析棄番風險
  - 輸入: `{ username: string, task_id?: string }`
  - 返回: `{ username, dropped_count, dropped_list[], watching_list[], planning_list[], model_metrics, drop_patterns }`

**後端引擎**:
- `HybridDropPredictionEngine`
  - `BERTRecommender` (20%): 序列預測（不符合觀看模式 = 高棄番風險）
  - `DropAnalysisEngine` (XGBoost 80%): 特徵預測

**XGBoost 特徵**:
- 動漫基本特徵: `episodes`, `average_score`, `popularity`, `season`
- 類型特徵: `genres` (multi-label)
- 標籤特徵: `tags` (multi-label, top 30)
- 製作公司: `studio` (label encoded)
- **新增特徵**:
  - `progress_ratio`: 觀看進度比例（已看集數/總集數）
  - `studio_drop_rate`: 使用者對該製作公司的歷史棄番率
  - `studio_watch_count`: 使用者看過該製作公司的作品數

**資料流程**:
1. 前端發送使用者名稱
2. 後端驗證使用者存在
3. 抓取並儲存使用者動漫清單
4. 訓練個人化 XGBoost 模型
5. 初始化混合預測引擎（可選啟用 BERT）
6. 對 `CURRENT` 和 `PLANNING` 狀態的動漫進行預測
7. 分析棄番模式（類型、標籤、製作公司統計）
8. 返回完整分析結果

**資料庫表**:
- `User` - 使用者資料
- `UserRating` - 使用者評分記錄
- `Anime` - 動漫資料

---

### 5. Recap.tsx - 年度回顧
**功能**: 生成使用者的年度動漫觀看統計

**後端 API**:
- `POST /recap` - 生成回顧報告
  - 輸入: `{ username: string, year?: number }`
  - 返回: `{ username, year, stats, top_genres[], top_studios[], timeline_data[], top_anime[], monthly_stats }`

**統計項目**:
- 總觀看數量
- 完成數量
- 總觀看時間
- 平均評分
- 最喜歡的類型
- 最喜歡的製作公司
- 月份觀看分布
- 時間軸數據

**資料流程**:
1. 前端發送使用者名稱和年份（可選）
2. 後端調用 `anilist_client.get_user_anime_list()` 獲取清單
3. 根據年份過濾資料
4. 計算各項統計數據
5. 生成時間軸和月份分布
6. 返回完整報告

**外部 API**:
- `https://graphql.anilist.co` - AniList GraphQL API
  - 獲取使用者動漫清單（包含完成時間、更新時間等）

**使用的組件**:
- `RecapStats` - 統計數字展示
- `TopGenres` - 最愛類型圖表
- `TopStudios` - 最愛製作公司
- `MonthlyChart` - 月份觀看分布圖
- `TopAnime` - 最高評分動漫列表

---

### 6. Timeline.tsx - 出生年份時間軸
**功能**: 展示使用者出生年份相關的動漫時間軸

**後端 API**:
- `POST /timeline` - 生成時間軸
  - 輸入: `{ username: string, birth_year: number }`
  - 返回: `{ username, birth_year, chronological_data[], timeline_data[], stats, birth_year_anime[] }`

**時間軸數據**:
- 按年份排序的觀看記錄
- 出生年份時的當季動漫
- 年齡對應的觀看統計

**資料流程**:
1. 前端發送使用者名稱和出生年份
2. 後端獲取使用者動漫清單
3. 計算每部動漫播放時使用者的年齡
4. 獲取出生年份當年的季番
5. 生成時間軸資料
6. 返回完整時間軸

**使用的組件**:
- `TimelineChart` - 時間軸視覺化
- `AgeStats` - 年齡統計
- `BirthYearAnime` - 出生年份動漫列表

---

### 7. Synergy.tsx - 觀看契合度分析
**功能**: 分析兩個使用者的觀看契合度

**後端 API**:
- `POST /synergy` - 分析契合度
  - 輸入: `{ username1: string, username2: string }`
  - 返回: `{ username1, username2, synergy_score, common_anime[], recommendations_for_1[], recommendations_for_2[], stats }`

**契合度計算**:
- 共同觀看的動漫數量
- 評分相似度
- 類型偏好相似度
- 製作公司偏好相似度

**推薦邏輯**:
- 找出對方喜歡但自己沒看過的動漫
- 根據契合度排序推薦

**資料流程**:
1. 前端發送兩個使用者名稱
2. 後端分別獲取兩個使用者的動漫清單
3. 計算共同觀看的動漫
4. 計算評分相似度
5. 分析偏好相似度
6. 生成互相推薦列表
7. 返回完整分析結果

**外部 API**:
- `https://graphql.anilist.co` - AniList GraphQL API
  - 獲取兩個使用者的動漫清單

**使用的組件**:
- `SynergyScore` - 契合度分數展示
- `CommonAnime` - 共同觀看動漫列表
- `MutualRecommendations` - 互相推薦列表
- `PreferenceComparison` - 偏好比較圖表

---

## 共用模組

### AniListClient (`anilist_client.py`)
**功能**: 與 AniList API 交互的客戶端

**主要方法**:
- `get_user_profile(username)` - 獲取使用者基本資料
- `get_user_anime_list(username)` - 獲取使用者動漫清單
- `get_seasonal_anime(season, year)` - 獲取季番
- `get_anime_details(anime_id)` - 獲取動漫詳細資料

**快取機制**:
- 使用 SQLite 資料庫快取 API 回應
- 減少對 AniList API 的請求次數
- 避免觸發 rate limit

---

### Ingest Data (`ingest_data.py`)
**功能**: 將 AniList 資料儲存到本地資料庫

**主要函數**:
- `fetch_and_store_user_data(session, username, progress_tracker)` - 儲存使用者資料
  - 創建或更新 `User` 記錄
  - 儲存 `UserRating` 記錄
  - 儲存對應的 `Anime` 記錄
  - 支援進度追蹤

- `fetch_and_store_anime(session, year, season)` - 儲存季番資料
  - 批次儲存動漫資料

**批次處理**:
- 每 50 筆 commit 一次
- 避免重複儲存已存在的資料

---

### Recommendation Engines

#### 1. BERT Recommender (`bert_model/bert_recommender_optimized.py`)
**功能**: 基於 BERT4Rec 的序列推薦

**訓練資料**:
- 使用全域使用者的觀看序列
- 預訓練模型位於 `bert_model/trained_models/`

**推薦邏輯**:
- 輸入: 使用者的觀看序列 (anime IDs)
- 輸出: 推薦分數最高的動漫列表
- 使用遮罩語言模型預測下一個可能觀看的動漫

**優化**:
- 支援 GPU 加速 (CUDA/MPS)
- 批次處理
- 快取機制

#### 2. Collaborative Recommendation (`recommendation_engine.py`)
**功能**: 基於協同過濾的推薦

**推薦邏輯**:
- 找出相似使用者（評分模式相似）
- 推薦相似使用者喜歡的動漫
- 考慮評分差異和觀看數量

#### 3. Hybrid Recommendation Engine (`hybrid_recommendation_engine.py`)
**功能**: 混合推薦引擎

**權重分配**:
- BERT: 80%
- Collaborative: 20%

**優點**:
- 結合序列模式和協同過濾
- 更準確的推薦結果

---

### Drop Prediction Engines

#### 1. Drop Analysis Engine (`drop_analysis_engine.py`)
**功能**: 基於 XGBoost 的棄番預測

**特徵工程**:
- 動漫特徵: episodes, average_score, popularity
- 類型特徵: genres (one-hot encoding)
- 標籤特徵: tags (one-hot encoding, top 30)
- 製作公司: studio (label encoding)
- 季節: season (one-hot encoding)
- **使用者特徵**:
  - `progress_ratio`: 觀看進度比例
  - `studio_drop_rate`: 對該製作公司的棄番率
  - `studio_watch_count`: 看過該製作公司的作品數

**訓練**:
- 使用使用者的 DROPPED 和 COMPLETED 記錄
- 處理類別不平衡 (scale_pos_weight)
- 個人化模型（每個使用者獨立訓練）

**預測原因生成**:
- 分析製作公司經驗
- 分析類型偏好
- 找出相似的已棄番動漫
- 考慮觀看進度（新增）

#### 2. Hybrid Drop Prediction Engine (`hybrid_drop_prediction_engine.py`)
**功能**: 混合棄番預測引擎

**權重分配**:
- XGBoost: 80%
- BERT: 20%

**BERT 在棄番預測的邏輯**:
- BERT 推薦分數低 = 不符合觀看模式 = 高棄番風險
- 棄番分數 = 1 - BERT 推薦分數

**超時保護**:
- BERT 預測設有 10 秒超時
- 使用 ThreadPoolExecutor 實現跨平台支援

---

## 資料庫架構

### 主資料庫 (SQLite: `anime.db`)

#### GlobalUser 表
- `id`: 主鍵
- `anilist_username`: AniList 使用者名稱
- `anilist_id`: AniList ID (唯一)
- `avatar`: 頭像 URL
- `created_at`: 創建時間

#### QuickID 表
- `id`: 主鍵
- `owner_id`: 外鍵 -> GlobalUser.id
- `anilist_username`: 常用 ID 的使用者名稱
- `anilist_id`: 常用 ID 的 AniList ID
- `avatar`: 頭像 URL
- `nickname`: 暱稱（可選）
- `created_at`: 創建時間

#### User 表
- `id`: 主鍵
- `username`: 使用者名稱
- `anilist_id`: AniList ID (唯一)
- `created_at`: 創建時間

#### Anime 表
- `id`: 主鍵 (AniList anime ID)
- `title_romaji`: 羅馬拼音標題
- `title_english`: 英文標題
- `genres`: 類型（逗號分隔）
- `tags`: 標籤（逗號分隔）
- `studios`: 製作公司（逗號分隔）
- `average_score`: 平均分數
- `popularity`: 人氣
- `episodes`: 集數
- `season`: 播放季節
- `season_year`: 播放年份

#### UserRating 表
- `id`: 主鍵
- `user_id`: 外鍵 -> User.id
- `anime_id`: 外鍵 -> Anime.id
- `score`: 評分
- `status`: 觀看狀態 (COMPLETED, CURRENT, DROPPED, PLANNING, PAUSED)
- `progress`: 觀看進度（集數）
- `created_at`: 創建時間
- `updated_at`: 更新時間

### BERT 模型資料庫 (`bert.db`)
- 儲存 BERT 訓練和推薦相關的資料
- 使用者序列快取
- 推薦結果快取

---

## 環境變數

### 前端 (`.env`)
```
VITE_BACKEND_URL=http://localhost:8000
```

### 後端
- 無需額外環境變數
- 資料庫路徑在程式碼中指定

---

## 部署架構

### 開發環境
- 前端: Vite Dev Server (Port 5173)
- 後端: Uvicorn (Port 8000)
- 資料庫: SQLite (本地檔案)

### 生產環境建議
- 前端: 靜態檔案部署 (Vercel, Netlify)
- 後端: FastAPI + Uvicorn (Docker)
- 資料庫: PostgreSQL (推薦) 或繼續使用 SQLite

---

## API 請求流程範例

### 推薦系統流程
```
前端 (Recommend.tsx)
  ↓ POST /recommend { username: "TheT" }
後端 (main.py)
  ↓ anilist_client.get_user_anime_list()
AniList API
  ↓ 返回使用者清單
後端 (ingest_data.py)
  ↓ 儲存到資料庫 (User, UserRating, Anime)
後端 (hybrid_recommendation_engine.py)
  ├─ BERT Recommender (80%)
  │   └─ 基於觀看序列預測
  └─ Collaborative Recommender (20%)
      └─ 基於相似使用者推薦
  ↓ 混合結果
前端
  └─ 顯示推薦列表
```

### 棄番預測流程
```
前端 (DropPredict.tsx)
  ↓ POST /analyze_drops { username: "TheT" }
後端 (main.py)
  ↓ 驗證使用者
  ↓ 抓取並儲存資料
  ↓ 訓練個人化 XGBoost 模型
後端 (hybrid_drop_prediction_engine.py)
  ├─ XGBoost (80%)
  │   ├─ 特徵: episodes, score, genres, tags, studio
  │   ├─ 新特徵: progress_ratio, studio_drop_rate, studio_watch_count
  │   └─ 預測棄番機率
  └─ BERT (20%)
      └─ 序列不符合度 → 棄番風險
  ↓ 混合預測
  ↓ 分析棄番模式
前端
  └─ 顯示預測結果和分析
```

---

## 重要技術細節

### 1. 進度追蹤（已移除）
- 原本使用 SSE (Server-Sent Events) 實現即時進度更新
- 因為連接問題已移除，改為直接等待 API 回應

### 2. 快取策略
- AniList API 回應快取 24 小時
- BERT 推薦結果快取
- 減少外部 API 調用

### 3. 錯誤處理
- 前端: 使用 try-catch 捕獲錯誤並顯示友善訊息
- 後端: HTTPException 統一錯誤格式
- 日誌: 使用 Python logging 記錄詳細錯誤

### 4. 性能優化
- 批次資料庫操作
- 延遲初始化 BERT 模型
- 使用 ThreadPoolExecutor 防止阻塞
- 超時保護機制

---

## 未來優化方向

### 1. 資料庫
- 遷移到 PostgreSQL
- 添加索引優化查詢
- 實現資料庫遷移系統

### 2. 快取
- 使用 Redis 替代 SQLite 快取
- 實現分散式快取

### 3. 推薦系統
- 定期重新訓練 BERT 模型
- 添加更多特徵（季節、播放時間等）
- A/B 測試不同權重組合

### 4. 使用者體驗
- 添加載入動畫
- 實現離線支援
- 多語言支援

### 5. 監控
- API 請求監控
- 錯誤率追蹤
- 性能指標收集