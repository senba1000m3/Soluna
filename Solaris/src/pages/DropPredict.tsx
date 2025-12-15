import React, { useState } from "react";
import {
  AlertTriangle,
  Loader2,
  BarChart2,
  CheckCircle,
  XCircle,
  BrainCircuit,
  X,
  Info,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ProgressMonitor } from "../components/ProgressMonitor";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

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

interface DropPatternStat {
  name: string;
  dropped: number;
  completed: number;
  total: number;
  drop_rate: number;
}

interface DropPatterns {
  top_dropped_tags: DropPatternStat[];
  top_dropped_genres: DropPatternStat[];
  top_dropped_studios: DropPatternStat[];
}

interface ModelMetrics {
  accuracy: number;
  sample_size: number;
  dropped_count: number;
  completed_count: number;
  top_features: [string, number][];
  error?: string;
}

interface AnalyzeResponse {
  task_id: string;
  username: string;
  dropped_count: number;
  dropped_list: AnimeItem[];
  watching_list: AnimeItem[];
  planning_list: AnimeItem[];
  model_metrics: ModelMetrics;
  drop_patterns: DropPatterns;
}

export const DropPredict = () => {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState("");
  const [modalAnime, setModalAnime] = useState<AnimeItem | null>(null);
  const [taskId, setTaskId] = useState<string>("");
  const [showProgress, setShowProgress] = useState(false);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) {
      setError("請輸入使用者名稱");
      return;
    }

    // Clear previous results and errors before starting new analysis
    setLoading(true);
    setError("");
    setResult(null);
    setShowProgress(true);

    // Generate a task ID
    const newTaskId = `drop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setTaskId(newTaskId);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze_drops`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: username.trim(),
          task_id: newTaskId,
        }),
      });

      if (!response.ok) {
        throw new Error("分析請求失敗，請確認 ID 是否正確");
      }

      const data: AnalyzeResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "發生錯誤，請稍後再試");
      setShowProgress(false);
    } finally {
      setLoading(false);
    }
  };

  const handleProgressComplete = () => {
    console.log("Progress completed");
    setShowProgress(false);
  };

  const handleProgressError = (error: string) => {
    console.error("Progress error:", error);
    setError(error);
    setShowProgress(false);
    setLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto px-4">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-red-500 to-orange-500 text-transparent bg-clip-text flex items-center justify-center gap-3">
          <AlertTriangle className="w-10 h-10 text-red-500" />
          棄番風險預測
        </h1>
        <p className="text-gray-400">
          分析你的棄番歷史，訓練 AI 模型，找出你的「棄番地雷區」
        </p>
      </div>

      <form
        onSubmit={handleAnalyze}
        className="mb-12 bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-xl max-w-2xl mx-auto"
      >
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="輸入 AniList ID (例如: senba1000m3)"
              className="flex-1 px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-red-500 focus:ring-2 focus:ring-red-500 outline-none transition-all text-lg"
            />
            <button
              type="submit"
              disabled={loading}
              className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-bold text-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  訓練模型中...
                </>
              ) : (
                <>
                  <BrainCircuit className="w-6 h-6" />
                  開始分析
                </>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-400 bg-gray-900/50 p-3 rounded-lg text-center">
            💡 每次分析都會自動從 AniList 抓取最新資料，所以你在 AniList
            上的任何變更都會被反映。
            <br />
            首次分析或資料變更較多時，可能需要幾秒鐘的時間。
          </p>
        </div>
      </form>

      {/* Progress Monitor */}
      {showProgress && taskId && (
        <div className="mb-8 max-w-2xl mx-auto">
          <ProgressMonitor
            taskId={taskId}
            onComplete={handleProgressComplete}
            onError={handleProgressError}
          />
        </div>
      )}

      {error && !showProgress && (
        <div className="text-center text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800 max-w-2xl mx-auto">
          {error}
        </div>
      )}

      {/* Modal for showing prediction reasons */}
      {modalAnime && (
        <div
          className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
          onClick={() => setModalAnime(null)}
        >
          <div
            className="bg-gray-800 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-gray-600 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="sticky top-0 bg-gray-800 border-b border-gray-700 p-6 flex items-start gap-4">
              <img
                src={modalAnime.cover}
                alt={modalAnime.title}
                className="w-20 h-28 object-cover rounded-lg"
              />
              <div className="flex-1">
                <h2 className="text-xl font-bold text-white mb-2">
                  {modalAnime.title}
                </h2>
                <div className="flex items-center gap-2">
                  <span className="text-gray-400 text-sm">棄番風險:</span>
                  <span
                    className={`text-2xl font-bold ${
                      modalAnime.drop_probability! > 0.7
                        ? "text-red-500"
                        : modalAnime.drop_probability! > 0.4
                          ? "text-yellow-500"
                          : "text-green-500"
                    }`}
                  >
                    {(modalAnime.drop_probability! * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <button
                onClick={() => setModalAnime(null)}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-6 h-6 text-gray-400" />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6">
              {/* Risk Bar */}
              <div>
                <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      modalAnime.drop_probability! > 0.7
                        ? "bg-red-500"
                        : modalAnime.drop_probability! > 0.4
                          ? "bg-yellow-500"
                          : "bg-green-500"
                    }`}
                    style={{
                      width: `${modalAnime.drop_probability! * 100}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>低風險</span>
                  <span>高風險</span>
                </div>
              </div>

              {/* Reasons */}
              {modalAnime.drop_reasons &&
                modalAnime.drop_reasons.length > 0 && (
                  <div>
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <Info className="w-5 h-5 text-blue-400" />
                      判斷依據
                    </h3>
                    <div className="space-y-3">
                      {modalAnime.drop_reasons.map((reason, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-900/50 p-4 rounded-lg border border-gray-700"
                        >
                          <div className="flex items-start gap-3">
                            <span className="text-blue-400 font-bold text-lg mt-0.5">
                              {idx + 1}.
                            </span>
                            <span className="text-gray-300 flex-1">
                              {reason}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              {/* Basic Info */}
              <div>
                <h3 className="text-lg font-bold text-white mb-3">作品資訊</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex">
                    <span className="text-gray-400 w-24">類型:</span>
                    <span className="text-gray-300">
                      {modalAnime.genres.join(", ")}
                    </span>
                  </div>
                  <div className="flex">
                    <span className="text-gray-400 w-24">進度:</span>
                    <span className="text-gray-300">
                      {modalAnime.progress} / {modalAnime.total_episodes || "?"}
                    </span>
                  </div>
                  {modalAnime.score > 0 && (
                    <div className="flex">
                      <span className="text-gray-400 w-24">評分:</span>
                      <span className="text-yellow-500 font-bold">
                        {modalAnime.score} 分
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-12 animate-fade-in">
          {/* Watching List with Predictions */}
          {result.watching_list.filter((a) => (a.drop_probability ?? 0) > 0)
            .length > 0 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-yellow-500" />
                正在觀看 - 棄番風險預測 (
                {
                  result.watching_list.filter(
                    (a) => (a.drop_probability ?? 0) > 0,
                  ).length
                }{" "}
                部)
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {result.watching_list
                  .filter((anime) => (anime.drop_probability ?? 0) > 0)
                  .map((anime) => (
                    <div
                      key={anime.id}
                      className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700 hover:border-yellow-500/50 transition-all shadow-lg group"
                    >
                      <div className="flex h-32">
                        <div className="w-24 flex-shrink-0">
                          <img
                            src={anime.cover}
                            alt={anime.title}
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="p-4 flex-1 flex flex-col justify-between">
                          <div className="space-y-2">
                            <h3 className="font-bold text-white line-clamp-2 group-hover:text-yellow-400 transition-colors">
                              {anime.title}
                            </h3>
                            <div>
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-400">棄番風險</span>
                                <span
                                  className={`font-bold ${
                                    anime.drop_probability! > 0.7
                                      ? "text-red-500"
                                      : anime.drop_probability! > 0.4
                                        ? "text-yellow-500"
                                        : "text-green-500"
                                  }`}
                                >
                                  {(anime.drop_probability! * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-2">
                                <div
                                  className={`h-full ${
                                    anime.drop_probability! > 0.7
                                      ? "bg-red-500"
                                      : anime.drop_probability! > 0.4
                                        ? "bg-yellow-500"
                                        : "bg-green-500"
                                  }`}
                                  style={{
                                    width: `${anime.drop_probability! * 100}%`,
                                  }}
                                />
                              </div>
                              {anime.drop_reasons &&
                                anime.drop_reasons.length > 0 && (
                                  <button
                                    onClick={() => setModalAnime(anime)}
                                    className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 transition-colors"
                                  >
                                    <Info className="w-3 h-3" />
                                    查看判斷依據
                                  </button>
                                )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Planning List with Predictions */}
          {result.planning_list.filter((a) => (a.drop_probability ?? 0) > 0)
            .length > 0 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-purple-500" />
                預定觀看 - 棄番風險預測 (
                {
                  result.planning_list.filter(
                    (a) => (a.drop_probability ?? 0) > 0,
                  ).length
                }{" "}
                部)
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {result.planning_list
                  .filter((anime) => (anime.drop_probability ?? 0) > 0)
                  .slice(0, 6)
                  .map((anime) => (
                    <div
                      key={anime.id}
                      className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700 hover:border-purple-500/50 transition-all shadow-lg group"
                    >
                      <div className="flex h-32">
                        <div className="w-24 flex-shrink-0">
                          <img
                            src={anime.cover}
                            alt={anime.title}
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="p-4 flex-1 flex flex-col justify-between">
                          <div className="space-y-2">
                            <h3 className="font-bold text-white line-clamp-2 group-hover:text-purple-400 transition-colors">
                              {anime.title}
                            </h3>
                            <div>
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-400">棄番風險</span>
                                <span
                                  className={`font-bold ${
                                    anime.drop_probability! > 0.7
                                      ? "text-red-500"
                                      : anime.drop_probability! > 0.4
                                        ? "text-yellow-500"
                                        : "text-green-500"
                                  }`}
                                >
                                  {(anime.drop_probability! * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-2">
                                <div
                                  className={`h-full ${
                                    anime.drop_probability! > 0.7
                                      ? "bg-red-500"
                                      : anime.drop_probability! > 0.4
                                        ? "bg-yellow-500"
                                        : "bg-green-500"
                                  }`}
                                  style={{
                                    width: `${anime.drop_probability! * 100}%`,
                                  }}
                                />
                              </div>
                              {anime.drop_reasons &&
                                anime.drop_reasons.length > 0 && (
                                  <button
                                    onClick={() => setModalAnime(anime)}
                                    className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 transition-colors"
                                  >
                                    <Info className="w-3 h-3" />
                                    查看判斷依據
                                  </button>
                                )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Model Metrics Section */}
          <div className="bg-gray-800 rounded-xl p-8 border border-gray-700 shadow-lg">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
              <BarChart2 className="w-6 h-6 text-blue-400" />
              AI 模型訓練報告
            </h2>

            {result.model_metrics.error ? (
              <div className="text-yellow-400 bg-yellow-900/20 p-4 rounded-lg">
                ⚠️ {result.model_metrics.error}
                <p className="text-sm mt-1 text-gray-400">
                  可能是因為你的棄番或完食紀錄太少，無法進行有效的機器學習訓練。
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Stats Cards */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                    <p className="text-gray-400 text-sm mb-1">模型準確率</p>
                    <p className="text-3xl font-bold text-green-400">
                      {(result.model_metrics.accuracy * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                    <p className="text-gray-400 text-sm mb-1">訓練樣本數</p>
                    <p className="text-3xl font-bold text-blue-400">
                      {result.model_metrics.sample_size}
                    </p>
                  </div>
                  <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                    <p className="text-gray-400 text-sm mb-1">完食作品</p>
                    <p className="text-3xl font-bold text-purple-400">
                      {result.model_metrics.completed_count}
                    </p>
                  </div>
                  <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                    <p className="text-gray-400 text-sm mb-1">棄番作品</p>
                    <p className="text-3xl font-bold text-red-400">
                      {result.model_metrics.dropped_count}
                    </p>
                  </div>
                </div>

                {/* Feature Importance Chart */}
                <div className="bg-gray-900/50 p-4 rounded-lg h-64">
                  <p className="text-gray-400 text-sm mb-4 text-center">
                    影響棄番的關鍵因素 (Top 5)
                  </p>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={result.model_metrics.top_features
                        .slice(0, 5)
                        .map(([name, val]) => ({
                          name: name.replace("Genre_", "").replace("Tag_", ""),
                          value: val,
                        }))}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="number" hide />
                      <YAxis
                        dataKey="name"
                        type="category"
                        width={100}
                        tick={{ fill: "#9CA3AF", fontSize: 12 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1F2937",
                          borderColor: "#374151",
                          color: "#F3F4F6",
                        }}
                      />
                      <Bar
                        dataKey="value"
                        fill="#F87171"
                        radius={[0, 4, 4, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {/* Drop Patterns Statistics */}
          {result.drop_patterns && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Top Dropped Tags */}
              {result.drop_patterns.top_dropped_tags.length > 0 && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-bold text-white mb-4">
                    🏷️ 最容易棄番的標籤
                  </h3>
                  <div className="space-y-3">
                    {result.drop_patterns.top_dropped_tags
                      .slice(0, 5)
                      .map((stat, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">{stat.name}</span>
                            <span className="text-red-400 font-bold">
                              {(stat.drop_rate * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex gap-2 text-xs text-gray-500">
                            <span>棄: {stat.dropped}</span>
                            <span>完: {stat.completed}</span>
                          </div>
                          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-red-500"
                              style={{ width: `${stat.drop_rate * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Top Dropped Genres */}
              {result.drop_patterns.top_dropped_genres.length > 0 && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-bold text-white mb-4">
                    🎭 最容易棄番的類型
                  </h3>
                  <div className="space-y-3">
                    {result.drop_patterns.top_dropped_genres
                      .slice(0, 5)
                      .map((stat, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">{stat.name}</span>
                            <span className="text-red-400 font-bold">
                              {(stat.drop_rate * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex gap-2 text-xs text-gray-500">
                            <span>棄: {stat.dropped}</span>
                            <span>完: {stat.completed}</span>
                          </div>
                          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-red-500"
                              style={{ width: `${stat.drop_rate * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Top Dropped Studios */}
              {result.drop_patterns.top_dropped_studios.length > 0 && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-bold text-white mb-4">
                    🏢 最容易棄番的製作公司
                  </h3>
                  <div className="space-y-3">
                    {result.drop_patterns.top_dropped_studios
                      .slice(0, 5)
                      .map((stat, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300 truncate">
                              {stat.name}
                            </span>
                            <span className="text-red-400 font-bold flex-shrink-0 ml-2">
                              {(stat.drop_rate * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex gap-2 text-xs text-gray-500">
                            <span>棄: {stat.dropped}</span>
                            <span>完: {stat.completed}</span>
                          </div>
                          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-red-500"
                              style={{ width: `${stat.drop_rate * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Dropped List Section */}
          <div>
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
              <XCircle className="w-6 h-6 text-red-500" />
              你的棄番黑歷史 ({result.dropped_count} 部)
            </h2>

            {result.dropped_list.length === 0 ? (
              <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
                <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">
                  太強了！你沒有棄番紀錄
                </h3>
                <p className="text-gray-400">
                  你是個有始有終的觀眾，或者你還沒開始整理清單？
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {result.dropped_list.map((anime) => (
                  <div
                    key={anime.id}
                    className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700 hover:border-red-500/50 transition-all shadow-lg group"
                  >
                    <div className="flex h-32">
                      <div className="w-24 flex-shrink-0">
                        <img
                          src={anime.cover}
                          alt={anime.title}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="p-4 flex-1 flex flex-col justify-between">
                        <div>
                          <h3 className="font-bold text-white line-clamp-2 mb-1 group-hover:text-red-400 transition-colors">
                            {anime.title}
                          </h3>
                          <div className="flex flex-wrap gap-1">
                            {anime.genres.slice(0, 2).map((g) => (
                              <span
                                key={g}
                                className="text-[10px] px-1.5 py-0.5 bg-gray-700 rounded text-gray-300"
                              >
                                {g}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="flex items-center justify-between text-xs text-gray-400 mt-2">
                          <span>
                            進度: {anime.progress} /{" "}
                            {anime.total_episodes || "?"}
                          </span>
                          {anime.score > 0 && (
                            <span className="text-yellow-500 font-bold">
                              {anime.score}分
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    {/* Progress Bar */}
                    <div className="h-1 bg-gray-700 w-full">
                      <div
                        className="h-full bg-red-500"
                        style={{
                          width: `${
                            anime.total_episodes
                              ? (anime.progress / anime.total_episodes) * 100
                              : 0
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
