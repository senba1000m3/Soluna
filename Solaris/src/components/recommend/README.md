# Recommend 組件

這個目錄包含了 Recommend（新番推薦系統）頁面使用的可重用組件。

## 組件列表

### AnimeCard
顯示推薦動畫的卡片組件，包含封面、標題、類型、評分等信息。

**Props:**
- `id: number` - 動畫 ID
- `title: AnimeTitle` - 動畫標題（romaji 和 english）
- `coverImage: AnimeCoverImage` - 封面圖片
- `genres: string[]` - 類型列表
- `averageScore: number` - 平均評分
- `popularity: number` - 人氣值
- `matchScore?: number` - 匹配分數（可選）
- `matchReasons?: MatchReason` - 匹配原因（可選）
- `onInfoClick?: () => void` - 點擊資訊按鈕的回調（可選）
- `onCardClick?: () => void` - 點擊卡片的回調（可選）

**使用範例:**
```tsx
<AnimeCard
  id={123}
  title={{ romaji: "Anime Title", english: "Anime Title" }}
  coverImage={{ large: "https://..." }}
  genres={["Action", "Fantasy"]}
  averageScore={85}
  popularity={50000}
  matchScore={92}
  matchReasons={matchReasons}
  onInfoClick={() => setSelectedAnime(anime)}
/>
```

**特點:**
- 自動顯示匹配分數徽章
- Hover 放大效果
- 點擊跳轉 AniList
- 資訊按鈕查看匹配原因

### MatchReasonModal
顯示動畫匹配原因的彈窗組件。

**Props:**
- `isOpen: boolean` - 是否顯示彈窗
- `onClose: () => void` - 關閉彈窗的回調
- `animeTitle: AnimeTitle` - 動畫標題
- `matchScore: number` - 匹配分數
- `matchReasons: MatchReason` - 匹配原因詳情

**使用範例:**
```tsx
<MatchReasonModal
  isOpen={selectedAnime !== null}
  onClose={() => setSelectedAnime(null)}
  animeTitle={selectedAnime.title}
  matchScore={selectedAnime.match_score}
  matchReasons={selectedAnime.match_reasons}
/>
```

**特點:**
- 顯示匹配度百分比
- 類型匹配權重可視化（進度條）
- 匹配原因說明
- 點擊背景關閉

### SeasonSelector
年份和季節選擇器組件。

**Props:**
- `year: string` - 當前選擇的年份
- `season: string` - 當前選擇的季節
- `onYearChange: (year: string) => void` - 年份變更回調
- `onSeasonChange: (season: string) => void` - 季節變更回調
- `nextSeason: { year, season, label }` - 下一季資訊（用於提示）

**使用範例:**
```tsx
<SeasonSelector
  year={year}
  season={season}
  onYearChange={setYear}
  onSeasonChange={setSeason}
  nextSeason={nextSeason}
/>
```

**特點:**
- 自動提示下一季資訊
- 響應式布局（桌面 2 列，手機 1 列）
- 季節選項：冬/春/夏/秋

## 類型定義

### AnimeTitle
```typescript
interface AnimeTitle {
  romaji: string;
  english: string | null;
}
```

### AnimeCoverImage
```typescript
interface AnimeCoverImage {
  large: string;
}
```

### MatchReason
```typescript
interface MatchReason {
  matched_genres: Array<{ genre: string; weight: number }>;
  total_weight: number;
  top_reason: string;
}
```

## 架構設計

這些組件採用模組化設計：
- 將 Recommend 頁面從 420 行減少到約 180 行
- 每個組件負責單一職責
- Props 類型安全，易於維護
- 樣式一致，使用 Tailwind CSS

## 工作流程

```
用戶輸入 → SeasonSelector → 發送請求
              ↓
          取得推薦結果
              ↓
    顯示 AnimeCard 列表
              ↓
    點擊資訊按鈕 → MatchReasonModal
```

## 樣式規範

### 顏色
- 主色調：Purple (`purple-600`, `purple-700`)
- 強調色：Pink (`pink-500`, `pink-600`)
- 背景：Gray (`gray-800`, `gray-900`)
- 邊框：Gray (`gray-700`)

### 圓角
- 卡片：`rounded-xl`
- 按鈕：`rounded-full` 或 `rounded-lg`
- 徽章：`rounded-full`

### 過渡效果
- Hover 縮放：`hover:scale-[1.02]`
- 顏色過渡：`transition-colors`
- 變形過渡：`transition-all duration-300`

## 與後端整合

### API 端點
```
POST /recommend
```

### 請求格式
```json
{
  "username": "senba1000m3",  // 可選
  "year": 2025,
  "season": "SPRING"
}
```

### 響應格式
```json
{
  "season": "SPRING",
  "year": 2025,
  "display_season": "春-4 月 (2025)",
  "recommendations": [
    {
      "id": 123,
      "title": { "romaji": "...", "english": "..." },
      "coverImage": { "large": "..." },
      "genres": ["Action", "Fantasy"],
      "averageScore": 85,
      "popularity": 50000,
      "match_score": 92,  // 如有提供 username
      "match_reasons": {  // 如有提供 username
        "matched_genres": [...],
        "total_weight": 5.2,
        "top_reason": "你喜歡 Action 和 Fantasy 類型"
      }
    }
  ]
}
```

## 最近更新

### 2024 年 12 月 - 初始重構
- 從 Recommend.tsx 提取組件
- 創建 AnimeCard、MatchReasonModal、SeasonSelector
- 統一類型定義和導出
- 代碼減少約 240 行（57%）

## 相關文件

- 主頁面：`src/pages/Recommend.tsx`
- 後端邏輯：`Lunaris/main.py` (recommend endpoint)
- 後端引擎：`Lunaris/recommendation_engine.py`

## 待優化

- [ ] 添加骨架屏載入狀態
- [ ] 支援無限滾動或分頁
- [ ] 添加篩選和排序功能
- [ ] 優化圖片載入（lazy loading）
- [ ] 添加動畫卡片收藏功能