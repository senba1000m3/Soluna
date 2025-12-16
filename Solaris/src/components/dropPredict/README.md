# DropPredict Components

這個目錄包含了「棄番風險預測」功能的所有 UI 組件。這些組件將原本龐大的 `DropPredict.tsx` 頁面拆分成可重用、易於維護的小組件。

## 📁 組件結構

```
dropPredict/
├── AnalysisForm.tsx          # 使用者輸入表單
├── AnimeCard.tsx              # 動漫卡片（支援多種變體）
├── AnimeDetailModal.tsx       # 動漫詳情模態框
├── AnimeListSection.tsx       # 動漫列表區域
├── ModelMetricsCard.tsx       # AI 模型指標卡片
├── DropPatternStats.tsx       # 棄番模式統計
├── index.ts                   # 統一導出
└── README.md                  # 本文件
```

## 🎯 組件說明

### 1. AnalysisForm

使用者輸入表單，用於輸入 AniList 使用者名稱並開始分析。

**Props:**
- `username: string` - 使用者名稱
- `onUsernameChange: (username: string) => void` - 使用者名稱變更回調
- `onSubmit: (e: React.FormEvent) => void` - 表單提交回調
- `loading: boolean` - 是否正在載入

**功能:**
- 整合 QuickIDSelector 快速選擇 ID
- 顯示提交按鈕（載入時顯示動畫）
- 顯示提示訊息

---

### 2. AnimeCard

通用動漫卡片組件，支援三種變體：watching、planning、dropped。

**Props:**
- `anime: AnimeItem` - 動漫資料
- `variant?: "watching" | "planning" | "dropped"` - 卡片變體（預設：watching）
- `onShowDetails?: (anime: AnimeItem) => void` - 顯示詳情回調

**變體差異:**
- **watching/planning**: 顯示棄番風險、風險進度條、判斷依據按鈕
- **dropped**: 顯示類型標籤、進度資訊、觀看進度條

**特色:**
- 根據風險等級動態顯示顏色（紅/黃/綠）
- Hover 效果和過渡動畫
- 響應式設計

---

### 3. AnimeDetailModal

顯示動漫詳細資訊和棄番風險分析的模態框。

**Props:**
- `anime: AnimeItem | null` - 要顯示的動漫（null 時不顯示）
- `onClose: () => void` - 關閉回調

**功能:**
- 顯示封面、標題、風險百分比
- 列出 AI 判斷依據（編號列表）
- 顯示基本資訊（類型、進度、評分）
- 點擊背景或 X 按鈕關閉
- 滾動支援（最大高度 80vh）

---

### 4. AnimeListSection

動漫列表區域組件，用於組織和顯示動漫卡片網格。

**Props:**
- `title: string` - 區域標題
- `animeList: AnimeItem[]` - 動漫列表
- `variant: "watching" | "planning" | "dropped"` - 卡片變體
- `icon?: "warning" | "x" | "check"` - 標題圖示
- `iconColor?: string` - 圖示顏色
- `emptyMessage?: string` - 空列表訊息
- `emptySubMessage?: string` - 空列表子訊息
- `onShowDetails?: (anime: AnimeItem) => void` - 顯示詳情回調
- `limit?: number` - 顯示數量限制

**功能:**
- 自動過濾有棄番風險的動漫（watching/planning）
- 空列表狀態處理
- 響應式網格佈局（1/2/3 列）
- 支援限制顯示數量

---

### 5. ModelMetricsCard

AI 模型訓練報告卡片，顯示模型性能指標。

**Props:**
- `metrics: ModelMetrics` - 模型指標資料

**功能:**
- 顯示 4 個關鍵指標（準確率、樣本數、完食/棄番數量）
- 特徵重要性橫向長條圖（Top 5）
- 錯誤狀態顯示
- 響應式佈局（桌面 2 列，行動裝置 1 列）

**指標說明:**
- **準確率**: 模型預測的準確度
- **訓練樣本數**: 用於訓練的總資料量
- **完食作品**: 完成觀看的動漫數量
- **棄番作品**: 中途放棄的動漫數量
- **關鍵因素**: 影響棄番決策的主要特徵（類型、標籤等）

---

### 6. DropPatternStats

棄番模式統計組件，顯示最容易棄番的標籤、類型、製作公司。

**Props:**
- `patterns: DropPatterns` - 棄番模式資料

**功能:**
- 3 個統計卡片（標籤、類型、製作公司）
- 每個卡片顯示 Top 5
- 顯示棄番率進度條
- 顯示棄番/完成數量
- 響應式網格佈局（1/3 列）

**PatternCard 子組件:**
- `title: string` - 卡片標題
- `emoji: string` - 標題 emoji
- `stats: DropPatternStat[]` - 統計資料

---

## 📦 類型定義

### AnimeItem
```typescript
interface AnimeItem {
  id: number;
  title: string;
  cover: string;
  score: number;
  progress: number;
  total_episodes: number | null;
  genres: string[];
  drop_probability?: number;
  drop_reasons?: string[];
}
```

### ModelMetrics
```typescript
interface ModelMetrics {
  accuracy: number;
  sample_size: number;
  dropped_count: number;
  completed_count: number;
  top_features: [string, number][];
  error?: string;
}
```

### DropPatternStat
```typescript
interface DropPatternStat {
  name: string;
  dropped: number;
  completed: number;
  total: number;
  drop_rate: number;
}
```

### DropPatterns
```typescript
interface DropPatterns {
  top_dropped_tags: DropPatternStat[];
  top_dropped_genres: DropPatternStat[];
  top_dropped_studios: DropPatternStat[];
}
```

---

## 🎨 設計模式

### 顏色系統

**風險等級顏色:**
- 🔴 高風險（> 70%）: `text-red-500` / `bg-red-500`
- 🟡 中風險（40-70%）: `text-yellow-500` / `bg-yellow-500`
- 🟢 低風險（< 40%）: `text-green-500` / `bg-green-500`

**卡片變體顏色:**
- **watching**: 黃色邊框 `border-yellow-500/50`
- **planning**: 紫色邊框 `border-purple-500/50`
- **dropped**: 紅色邊框 `border-red-500/50`

### 響應式設計

所有組件都採用 Tailwind 的響應式類別：
- `md:` - 平板尺寸（768px+）
- `lg:` - 桌面尺寸（1024px+）

網格佈局通常為：
- 行動裝置: 1 列
- 平板: 2 列
- 桌面: 3 列

---

## 🔄 使用範例

### 基本使用

```typescript
import {
  AnalysisForm,
  AnimeListSection,
  ModelMetricsCard,
  DropPatternStats,
  AnimeDetailModal,
} from "../components/dropPredict";

function DropPredict() {
  const [username, setUsername] = useState("");
  const [modalAnime, setModalAnime] = useState<AnimeItem | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  return (
    <div>
      <AnalysisForm
        username={username}
        onUsernameChange={setUsername}
        onSubmit={handleAnalyze}
        loading={loading}
      />

      {result && (
        <>
          <AnimeListSection
            title="正在觀看 - 棄番風險預測"
            animeList={result.watching_list}
            variant="watching"
            onShowDetails={setModalAnime}
          />

          <ModelMetricsCard metrics={result.model_metrics} />

          <DropPatternStats patterns={result.drop_patterns} />
        </>
      )}

      <AnimeDetailModal
        anime={modalAnime}
        onClose={() => setModalAnime(null)}
      />
    </div>
  );
}
```

---

## 🧪 測試建議

### 單元測試
- 測試各組件的渲染
- 測試 props 傳遞和回調觸發
- 測試條件渲染（空列表、錯誤狀態等）

### 整合測試
- 測試完整的使用者流程
- 測試 API 回應的資料顯示
- 測試模態框的開啟/關閉

### 視覺回歸測試
- 測試不同風險等級的顏色顯示
- 測試響應式佈局
- 測試 Hover 效果和動畫

---

## 🚀 未來改進

1. **性能優化**
   - 使用 React.memo 優化不必要的重渲染
   - 虛擬滾動大列表（如果動漫數量很多）
   - 圖片懶加載

2. **功能增強**
   - 排序和篩選選項
   - 匯出報告功能
   - 分享功能

3. **可訪問性**
   - 鍵盤導航支援
   - ARIA 標籤
   - 螢幕閱讀器支援

4. **測試覆蓋**
   - 添加 Jest + React Testing Library 測試
   - 添加 Storybook 用於組件開發

---

## 📝 維護注意事項

1. **類型安全**: 所有組件都有完整的 TypeScript 類型定義
2. **依賴管理**: 使用 recharts 進行圖表渲染，確保版本兼容
3. **樣式一致性**: 使用 Tailwind 類別保持樣式一致
4. **組件獨立性**: 每個組件應該可以獨立使用和測試

---

## 🔗 相關文件

- [DropPredict 頁面重構文檔](../../../../DROP_PREDICT_REFACTOR.md)
- [Refactor 計劃](../../../../REFACTOR_PLAN.md)
- [Timeline 組件範例](../timeline/README.md)
- [Recommend 組件範例](../recommend/README.md)