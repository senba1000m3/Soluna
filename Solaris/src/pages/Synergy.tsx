import React, { useState } from "react";
import {
  Users,
  Heart,
  Zap,
  Loader2,
  AlertCircle,
  TrendingUp,
  BarChart3,
  Star,
  Film,
  Target,
} from "lucide-react";
import {
  StatItem,
  TabButton,
  RecommendationCard,
  UserAvatar,
  DisagreementsModal,
  GenreRadarChart,
} from "../components/synergy";
import { SynergyResponse, SynergyTab, CommonAnime } from "../types/synergy";
import {
  getScoreColor,
  getScoreMessage,
  formatScore,
  shouldShowScoreDiff,
  getAnimeUrl,
  getUserUrl,
  hasRecommendations,
  shouldShowViewMore,
} from "../utils/synergyHelpers";
import { QuickIDSelector } from "../components/QuickIDSelector";
import { BACKEND_URL } from "../config/env";

/**
 * 共鳴配對頁面
 *
 * 功能概述：
 * 1. 比較兩位 AniList 用戶的動畫品味契合度
 * 2. 展示共同觀看的作品、類型偏好、評分差異等
 * 3. 提供互相推薦功能
 *
 * 評分機制：
 * - 自動統一為 0-10.0 分制（兼容 0-100 和 0-10 兩種系統）
 * - 共鳴指數使用 Cosine Similarity 計算（0-100%）
 * - 評分差異以絕對值表示，保留兩位小數
 *
 * 雷達圖計算：
 * - 基於用戶的類型偏好權重（考慮評分和觀看頻率）
 * - 正規化到 0-100 範圍
 * - 選取最重要的 8 個類型展示
 * - 確保所有值非負數
 */
export const Synergy = () => {
  const [user1, setUser1] = useState("");
  const [user2, setUser2] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<SynergyResponse | null>(null);
  const [activeTab, setActiveTab] = useState<SynergyTab>("overview");
  const [showDisagreementsModal, setShowDisagreementsModal] = useState(false);

  /**
   * 處理比較請求
   */
  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();

    // 驗證輸入
    if (!isValidInput(user1, user2)) {
      setError("請輸入兩個使用者名稱");
      return;
    }

    // 開始請求
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await fetchComparison(user1.trim(), user2.trim());
      setResult(data);
      setActiveTab("overview");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "發生錯誤，請確認使用者名稱是否正確");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* 頁面標題 */}
      <PageHeader />

      {/* 輸入表單 */}
      <ComparisonForm
        user1={user1}
        user2={user2}
        loading={loading}
        onUser1Change={setUser1}
        onUser2Change={setUser2}
        onSubmit={handleCompare}
      />

      {/* 錯誤提示 */}
      {error && <ErrorMessage message={error} />}

      {/* 結果展示 */}
      {result && (
        <div className="animate-fade-in space-y-8">
          {/* 配對分數卡片 */}
          <CompatibilityCard result={result} />

          {/* 統計對比 */}
          <StatisticsComparison result={result} />

          {/* 快速指標 */}
          <QuickMetrics result={result} />

          {/* 標籤導航 */}
          <TabNavigation
            activeTab={activeTab}
            onTabChange={setActiveTab}
            result={result}
          />

          {/* 標籤內容 */}
          <TabContent
            activeTab={activeTab}
            result={result}
            onShowDisagreements={() => setShowDisagreementsModal(true)}
          />
        </div>
      )}

      {/* 品味分歧彈窗 */}
      {result && (
        <DisagreementsModal
          isOpen={showDisagreementsModal}
          onClose={() => setShowDisagreementsModal(false)}
          disagreements={result.disagreements}
          user1={result.user1}
          user2={result.user2}
        />
      )}
    </div>
  );
};

/**
 * 驗證輸入是否有效
 */
const isValidInput = (user1: string, user2: string): boolean => {
  return user1.trim() !== "" && user2.trim() !== "";
};

/**
 * 發送比較請求到後端
 */
const fetchComparison = async (
  user1: string,
  user2: string,
): Promise<SynergyResponse> => {
  const response = await fetch(`${BACKEND_URL}/pair_compare`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user1, user2 }),
  });

  if (!response.ok) {
    const errData = await response.json();
    throw new Error(errData.detail || "比較請求失敗");
  }

  return await response.json();
};

/**
 * 頁面標題組件
 */
const PageHeader: React.FC = () => (
  <div className="text-center mb-12">
    <h1 className="text-5xl font-bold mb-4 mt-5 bg-gradient-to-r from-blue-400 to-teal-400 text-transparent bg-clip-text flex items-center justify-center gap-3">
      <Users className="w-10 h-10 text-blue-400" />
      共鳴配對
    </h1>
    <p className="text-gray-400">
      輸入兩個 AniList ID，分析你們的動畫品味契合度
    </p>
  </div>
);

/**
 * 比較表單組件
 */
const ComparisonForm: React.FC<{
  user1: string;
  user2: string;
  loading: boolean;
  onUser1Change: (value: string) => void;
  onUser2Change: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
}> = ({ user1, user2, loading, onUser1Change, onUser2Change, onSubmit }) => (
  <form
    onSubmit={onSubmit}
    className="mb-12 bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-xl"
  >
    <div className="flex flex-col md:flex-row gap-6 items-center justify-center mb-8">
      <div className="w-full md:flex-1">
        <QuickIDSelector
          value={user1}
          onChange={onUser1Change}
          label="使用者 A"
          placeholder="例如：senba1000m3"
          required={true}
        />
      </div>

      <div className="flex items-center justify-center md:pt-2">
        <div className="bg-gray-700 p-3 rounded-full">
          <Zap className="w-6 h-6 text-yellow-400 fill-yellow-400" />
        </div>
      </div>

      <div className="w-full md:flex-1">
        <QuickIDSelector
          value={user2}
          onChange={onUser2Change}
          label="使用者 B"
          placeholder="例如：momoha100m3"
          required={true}
        />
      </div>
    </div>

    <button
      type="submit"
      disabled={loading}
      className="w-full px-6 py-4 mb-1 bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 rounded-lg font-bold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
    >
      {loading ? (
        <>
          <Loader2 className="w-6 h-6 animate-spin" />
          正在分析共鳴...
        </>
      ) : (
        <>
          <Heart className="w-6 h-6" />
          開始配對分析
        </>
      )}
    </button>
  </form>
);

/**
 * 錯誤訊息組件
 */
const ErrorMessage: React.FC<{ message: string }> = ({ message }) => (
  <div className="flex items-center justify-center gap-2 text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800">
    <AlertCircle className="w-5 h-5" />
    {message}
  </div>
);

/**
 * 配對分數卡片組件
 */
const CompatibilityCard: React.FC<{ result: SynergyResponse }> = ({
  result,
}) => (
  <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 text-center relative overflow-hidden">
    <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-blue-500 to-teal-500" />

    <div className="flex flex-col md:flex-row items-center justify-center gap-8 mb-8">
      {/* 用戶1 */}
      <a
        href={getUserUrl(result.user1.name)}
        target="_blank"
        rel="noopener noreferrer"
        className="flex flex-col items-center hover:opacity-80 transition-opacity"
      >
        <img
          src={result.user1.avatar.large}
          alt={result.user1.name}
          className="w-24 h-24 rounded-full border-4 border-blue-500 shadow-lg mb-3"
        />
        <h3 className="text-xl font-bold">{result.user1.name}</h3>
      </a>

      {/* 共鳴指數 */}
      <div className="flex flex-col items-center justify-center px-8">
        <div className="text-6xl font-black mb-2 flex items-baseline gap-1">
          <span className={getScoreColor(result.compatibility_score)}>
            {result.compatibility_score.toFixed(1)}
          </span>
          <span className="text-2xl text-gray-500">%</span>
        </div>
        <div className="text-sm text-gray-400 uppercase tracking-widest font-semibold">
          共鳴指數
        </div>
      </div>

      {/* 用戶2 */}
      <a
        href={getUserUrl(result.user2.name)}
        target="_blank"
        rel="noopener noreferrer"
        className="flex flex-col items-center hover:opacity-80 transition-opacity"
      >
        <img
          src={result.user2.avatar.large}
          alt={result.user2.name}
          className="w-24 h-24 rounded-full border-4 border-teal-500 shadow-lg mb-3"
        />
        <h3 className="text-xl font-bold">{result.user2.name}</h3>
      </a>
    </div>

    <p className="text-xl text-blue-200 font-medium bg-blue-900/30 py-3 px-6 rounded-full inline-block">
      {getScoreMessage(result.compatibility_score)}
    </p>
  </div>
);

/**
 * 統計對比組件
 */
const StatisticsComparison: React.FC<{ result: SynergyResponse }> = ({
  result,
}) => (
  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    {/* 用戶1統計 */}
    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <UserAvatar
          username={result.user1.name}
          avatarUrl={result.user1.avatar.large}
          size="medium"
          borderColor="border-blue-500 hover:ring-blue-400"
        />
        <a
          href={getUserUrl(result.user1.name)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xl font-bold hover:text-blue-400 transition-colors"
        >
          {result.user1.name}
        </a>
      </div>
      <div className="space-y-4">
        <StatItem
          icon={<Film className="w-5 h-5" />}
          label="總動畫數"
          value={result.stats.user1.total_anime}
        />
        <StatItem
          icon={<Target className="w-5 h-5" />}
          label="已完成"
          value={result.stats.user1.completed}
        />
        <StatItem
          icon={<Star className="w-5 h-5" />}
          label="平均評分"
          value={result.stats.user1.avg_score.toFixed(1)}
        />
        <StatItem
          icon={<TrendingUp className="w-5 h-5" />}
          label="觀看集數"
          value={result.stats.user1.episodes_watched}
        />
      </div>
    </div>

    {/* 用戶2統計 */}
    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <UserAvatar
          username={result.user2.name}
          avatarUrl={result.user2.avatar.large}
          size="medium"
          borderColor="border-teal-500 hover:ring-teal-400"
        />
        <a
          href={getUserUrl(result.user2.name)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xl font-bold hover:text-teal-400 transition-colors"
        >
          {result.user2.name}
        </a>
      </div>
      <div className="space-y-4">
        <StatItem
          icon={<Film className="w-5 h-5" />}
          label="總動畫數"
          value={result.stats.user2.total_anime}
        />
        <StatItem
          icon={<Target className="w-5 h-5" />}
          label="已完成"
          value={result.stats.user2.completed}
        />
        <StatItem
          icon={<Star className="w-5 h-5" />}
          label="平均評分"
          value={result.stats.user2.avg_score.toFixed(1)}
        />
        <StatItem
          icon={<TrendingUp className="w-5 h-5" />}
          label="觀看集數"
          value={result.stats.user2.episodes_watched}
        />
      </div>
    </div>
  </div>
);

/**
 * 快速指標組件
 */
const QuickMetrics: React.FC<{ result: SynergyResponse }> = ({ result }) => (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
    <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-6 border border-purple-700">
      <div className="text-3xl font-bold mb-2">{result.common_count}</div>
      <div className="text-purple-200">共同觀看作品</div>
    </div>
    <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-6 border border-green-700">
      <div className="text-3xl font-bold mb-2">
        {result.common_genres.length}
      </div>
      <div className="text-green-200">共同喜好類型</div>
    </div>
    <div className="bg-gradient-to-br from-orange-900/50 to-orange-800/30 rounded-xl p-6 border border-orange-700">
      <div className="text-3xl font-bold mb-2">
        {result.avg_score_difference.toFixed(1)}
      </div>
      <div className="text-orange-200">平均評分差異</div>
    </div>
  </div>
);

/**
 * 標籤導航組件
 */
const TabNavigation: React.FC<{
  activeTab: SynergyTab;
  onTabChange: (tab: SynergyTab) => void;
  result: SynergyResponse;
}> = ({ activeTab, onTabChange, result }) => (
  <div className="flex gap-2 overflow-x-auto pb-2">
    <TabButton
      active={activeTab === "overview"}
      onClick={() => onTabChange("overview")}
      icon={<BarChart3 className="w-5 h-5" />}
      label="品味分析"
    />
    <TabButton
      active={activeTab === "anime"}
      onClick={() => onTabChange("anime")}
      icon={<Film className="w-5 h-5" />}
      label={`共同作品 (${result.common_count})`}
    />
    <TabButton
      active={activeTab === "recommendations"}
      onClick={() => onTabChange("recommendations")}
      icon={<Heart className="w-5 h-5" />}
      label="互相推薦"
    />
    <TabButton
      active={activeTab === "disagreements"}
      onClick={() => onTabChange("disagreements")}
      icon={<AlertCircle className="w-5 h-5" />}
      label="品味分歧"
    />
  </div>
);

/**
 * 標籤內容組件
 */
const TabContent: React.FC<{
  activeTab: SynergyTab;
  result: SynergyResponse;
  onShowDisagreements: () => void;
}> = ({ activeTab, result, onShowDisagreements }) => {
  switch (activeTab) {
    case "overview":
      return <OverviewTab result={result} />;
    case "anime":
      return <CommonAnimeTab result={result} />;
    case "recommendations":
      return <RecommendationsTab result={result} />;
    case "disagreements":
      return (
        <DisagreementsTab result={result} onShowAll={onShowDisagreements} />
      );
    default:
      return null;
  }
};

/**
 * 品味分析標籤
 */
const OverviewTab: React.FC<{ result: SynergyResponse }> = ({ result }) => (
  <div className="space-y-8">
    {/* 雷達圖 */}
    <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
      <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <BarChart3 className="w-6 h-6 text-purple-400" />
        類型偏好雷達圖
      </h3>
      <GenreRadarChart
        radarData={result.radar_data}
        user1Name={result.user1.name}
        user2Name={result.user2.name}
      />
    </div>

    {/* 共同類型 */}
    {result.common_genres.length > 0 && (
      <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Zap className="w-6 h-6 text-yellow-400" />
          共同喜好領域
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.common_genres.map((genre, idx) => (
            <div
              key={genre.genre}
              className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <span className="text-gray-400 font-mono text-lg">
                  #{idx + 1}
                </span>
                <span className="font-bold text-lg">{genre.genre}</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="px-3 py-1 bg-gradient-to-r from-blue-500 to-teal-500 rounded-full text-sm font-bold">
                  {genre.similarity.toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);

/**
 * 共同作品標籤
 */
const CommonAnimeTab: React.FC<{ result: SynergyResponse }> = ({ result }) => (
  <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
    <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
      <Film className="w-6 h-6 text-blue-400" />
      共同觀看的動畫 ({result.common_count})
    </h3>
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
      {result.common_anime.map((anime) => (
        <CommonAnimeCard key={anime.id} anime={anime} />
      ))}
    </div>
  </div>
);

/**
 * 共同作品卡片
 */
const CommonAnimeCard: React.FC<{ anime: CommonAnime }> = ({ anime }) => (
  <a
    href={getAnimeUrl(anime.id)}
    target="_blank"
    rel="noopener noreferrer"
    className="bg-gray-700/50 rounded-lg overflow-hidden hover:ring-2 hover:ring-blue-500 transition-all cursor-pointer group block"
  >
    <img
      src={anime.coverImage}
      alt={anime.title}
      className="w-full h-48 object-cover group-hover:scale-105 transition-transform"
    />
    <div className="p-3">
      <h4 className="font-semibold text-sm mb-3 line-clamp-2 h-10">
        {anime.title}
      </h4>
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full border-2 border-blue-500" />
          <span
            className={`font-bold ${anime.user1_score > 0 ? "" : "text-gray-500"}`}
          >
            {formatScore(anime.user1_score)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full border-2 border-teal-500" />
          <span
            className={`font-bold ${anime.user2_score > 0 ? "" : "text-gray-500"}`}
          >
            {formatScore(anime.user2_score)}
          </span>
        </div>
      </div>
      {shouldShowScoreDiff(anime.user1_score, anime.user2_score) && (
        <div className="mt-2 text-xs text-gray-400 text-center">
          差異: {anime.score_diff.toFixed(2)}
        </div>
      )}
      {(!anime.user1_score || !anime.user2_score) && (
        <div className="mt-2 text-xs text-gray-500 text-center">未評分</div>
      )}
    </div>
  </a>
);

/**
 * 推薦標籤
 */
const RecommendationsTab: React.FC<{ result: SynergyResponse }> = ({
  result,
}) => (
  <div className="space-y-8">
    {/* 推薦給用戶2 */}
    <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
      <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Heart className="w-6 h-6 text-blue-400" />
        推薦給 {result.user2.name}
        <span className="text-sm text-gray-400 font-normal ml-2">
          ({result.user1.name} 的高分作品)
        </span>
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {hasRecommendations(result.recommendations.for_user2) ? (
          result.recommendations.for_user2.map((anime) => (
            <RecommendationCard key={anime.id} anime={anime} />
          ))
        ) : (
          <div className="col-span-full text-center text-gray-400 py-8">
            暫無推薦（需要高分作品）
          </div>
        )}
      </div>
    </div>

    {/* 推薦給用戶1 */}
    <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
      <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Heart className="w-6 h-6 text-teal-400" />
        推薦給 {result.user1.name}
        <span className="text-sm text-gray-400 font-normal ml-2">
          ({result.user2.name} 的高分作品)
        </span>
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {hasRecommendations(result.recommendations.for_user1) ? (
          result.recommendations.for_user1.map((anime) => (
            <RecommendationCard key={anime.id} anime={anime} />
          ))
        ) : (
          <div className="col-span-full text-center text-gray-400 py-8">
            暫無推薦（需要高分作品）
          </div>
        )}
      </div>
    </div>
  </div>
);

/**
 * 品味分歧標籤
 */
const DisagreementsTab: React.FC<{
  result: SynergyResponse;
  onShowAll: () => void;
}> = ({ result, onShowAll }) => (
  <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
    <div className="flex items-center justify-between mb-6">
      <h3 className="text-2xl font-bold flex items-center gap-2">
        <AlertCircle className="w-6 h-6 text-orange-400" />
        品味分歧最大的作品
      </h3>
      {shouldShowViewMore(result.disagreements) && (
        <button
          onClick={onShowAll}
          className="px-4 py-2 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-lg font-semibold transition-all flex items-center gap-2 shadow-lg"
        >
          查看全部 ({result.disagreements.length})
          <AlertCircle className="w-4 h-4" />
        </button>
      )}
    </div>
    <div className="space-y-4">
      {result.disagreements.length > 0 ? (
        result.disagreements
          .slice(0, 5)
          .map((anime, idx) => (
            <DisagreementPreviewItem
              key={anime.id}
              anime={anime}
              index={idx}
              user1={result.user1}
              user2={result.user2}
            />
          ))
      ) : (
        <div className="text-center py-8 text-gray-400">
          <p>雙方評分非常一致，沒有明顯的品味分歧！</p>
        </div>
      )}
    </div>
  </div>
);

/**
 * 品味分歧預覽項目
 */
const DisagreementPreviewItem: React.FC<{
  anime: CommonAnime;
  index: number;
  user1: any;
  user2: any;
}> = ({ anime, index, user1, user2 }) => (
  <div className="bg-gray-700/50 rounded-lg p-4 flex flex-col md:flex-row gap-4 items-start md:items-center">
    <div className="text-2xl font-bold text-gray-500 w-8">#{index + 1}</div>
    <a href={getAnimeUrl(anime.id)} target="_blank" rel="noopener noreferrer">
      <img
        src={anime.coverImage}
        alt={anime.title}
        className="w-16 h-24 object-cover rounded hover:ring-2 hover:ring-orange-400 transition-all"
      />
    </a>
    <div className="flex-1">
      <a
        href={getAnimeUrl(anime.id)}
        target="_blank"
        rel="noopener noreferrer"
        className="font-bold text-lg mb-2 hover:text-blue-400 transition-colors block"
      >
        {anime.title}
      </a>
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <UserAvatar
            username={user1.name}
            avatarUrl={user1.avatar.large}
            size="small"
            borderColor="border-blue-500 hover:ring-blue-400"
          />
          <div>
            <div className="text-xs text-gray-400">{user1.name}</div>
            <div className="text-xl font-bold text-blue-400">
              {formatScore(anime.user1_score, 1)}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <UserAvatar
            username={user2.name}
            avatarUrl={user2.avatar.large}
            size="small"
            borderColor="border-teal-500 hover:ring-teal-400"
          />
          <div>
            <div className="text-xs text-gray-400">{user2.name}</div>
            <div className="text-xl font-bold text-teal-400">
              {formatScore(anime.user2_score, 1)}
            </div>
          </div>
        </div>
        <div className="ml-auto">
          <div className="text-xs text-gray-400">評分差異</div>
          <div className="text-2xl font-bold text-orange-400">
            {anime.score_diff.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  </div>
);
