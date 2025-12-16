# Timeline 組件

這個目錄包含了 Timeline（動畫大世紀）頁面使用的可重用組件。

## 組件列表

### StatCard
顯示單個統計數據的卡片組件。

**Props:**
- `label: string` - 統計項目的標籤
- `value: string | number` - 統計數值
- `subtitle?: string` - 副標題（可選）

**使用範例:**
```tsx
<StatCard
  label="動畫成癮年"
  value={2023}
  subtitle="看了 42 部"
/>
```

### StatsPanel
統一顯示所有統計數據的面板組件。

**Props:**
- `stats: TimelineStats` - 包含所有統計數據的對象

**統計數據包含:**
1. `most_active_year` - 看最多動畫的年份
2. `favorite_genre` - 最喜歡的類型
3. `total_watch_time` - 總觀看時間
4. `favorite_season` - 今年追番最多的季節（新增）
5. `favorites_count` - 喜歡的作品數（評分 8 分以上，新增）

**使用範例:**
```tsx
<StatsPanel stats={result.stats} />
```

### BirthYearCard
顯示用戶出生年份的霸權動畫列表。

**Props:**
- `birthYear: number` - 出生年份
- `animeList: BirthYearAnime[]` - 該年份的動畫列表

**使用範例:**
```tsx
<BirthYearCard
  birthYear={2000}
  animeList={result.birth_year_anime}
/>
```

## 架構設計

這些組件採用了與 Synergy 頁面相同的模組化架構：
- 將大型頁面拆分成小型、可重用的組件
- 每個組件負責單一職責
- 通過 `index.ts` 統一導出，簡化導入語句

## 最近更新

### 2024 年更新
- 新增 `favorite_season` 統計：顯示今年（current year）看最多動畫的季節
- 新增 `favorites_count` 統計：計算評分 8 分（含）以上的作品數量
- 重構 Timeline 頁面，將內嵌的統計顯示邏輯提取為獨立組件
- 創建模組化的組件結構，提高代碼可維護性

## 相關文件

- 前端頁面: `src/pages/Timeline.tsx`
- 後端邏輯: `Lunaris/recommendation_engine.py` (calculate_timeline_stats 方法)
- 類型定義: 直接在 `Timeline.tsx` 中的 `TimelineStats` 接口