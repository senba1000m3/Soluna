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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface UserProfile {
  name: string;
  avatar: {
    large: string;
  };
}

interface CommonGenre {
  genre: string;
  score: number;
  similarity: number;
}

interface CommonAnime {
  id: number;
  title: string;
  coverImage: string;
  user1_score: number;
  user2_score: number;
  score_diff: number;
  average_score: number;
  both_rated: boolean;
}

interface RadarData {
  labels: string[];
  user1: number[];
  user2: number[];
}

interface UserStats {
  total_anime: number;
  completed: number;
  avg_score: number;
  episodes_watched: number;
}

interface Recommendation {
  id: number;
  title: string;
  coverImage: string;
  score: number;
  user_scored: boolean;
  genres: string[];
}

interface SynergyResponse {
  user1: UserProfile;
  user2: UserProfile;
  compatibility_score: number;
  common_genres: CommonGenre[];
  common_anime: CommonAnime[];
  common_count: number;
  disagreements: CommonAnime[];
  avg_score_difference: number;
  radar_data: RadarData;
  stats: {
    user1: UserStats;
    user2: UserStats;
  };
  recommendations: {
    for_user1: Recommendation[];
    for_user2: Recommendation[];
  };
  message: string;
}

export const Synergy = () => {
  const [user1, setUser1] = useState("");
  const [user2, setUser2] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<SynergyResponse | null>(null);
  const [activeTab, setActiveTab] = useState<
    "overview" | "anime" | "recommendations" | "disagreements"
  >("overview");
  const [showDisagreementsModal, setShowDisagreementsModal] = useState(false);

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user1.trim() || !user2.trim()) {
      setError("請輸入兩個使用者名稱");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${BACKEND_URL}/pair_compare`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user1: user1.trim(),
          user2: user2.trim(),
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "比較請求失敗");
      }

      const data: SynergyResponse = await response.json();
      setResult(data);
      setActiveTab("overview");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "發生錯誤，請確認使用者名稱是否正確");
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-400";
    if (score >= 60) return "text-yellow-400";
    return "text-red-400";
  };

  const getScoreMessage = (score: number) => {
    if (score >= 90) return "靈魂伴侶！你們的品味驚人地相似！";
    if (score >= 80) return "非常合拍！有很多共同話題。";
    if (score >= 60) return "還不錯，有些共同喜好。";
    if (score >= 40) return "品味有些差異，但可以互相推坑。";
    return "水火不容？或許是互補的關係！";
  };

  const formatRadarData = (radarData: RadarData) => {
    return radarData.labels.map((label, idx) => ({
      subject: label,
      [result?.user1.name || "User 1"]: radarData.user1[idx],
      [result?.user2.name || "User 2"]: radarData.user2[idx],
    }));
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-teal-400 text-transparent bg-clip-text flex items-center justify-center gap-3">
          <Users className="w-10 h-10 text-blue-400" />
          共鳴配對
        </h1>
        <p className="text-gray-400">
          輸入兩個 AniList ID，分析你們的動畫品味契合度
        </p>
      </div>

      <form
        onSubmit={handleCompare}
        className="mb-12 bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-xl"
      >
        <div className="flex flex-col md:flex-row gap-6 items-center justify-center mb-8">
          <div className="w-full">
            <label className="block text-sm font-medium mb-2 text-gray-300">
              使用者 A
            </label>
            <input
              type="text"
              value={user1}
              onChange={(e) => setUser1(e.target.value)}
              placeholder="例如: senba1000m3"
              className="w-full px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-lg"
            />
          </div>

          <div className="flex items-center justify-center pt-6">
            <div className="bg-gray-700 p-3 rounded-full">
              <Zap className="w-6 h-6 text-yellow-400 fill-yellow-400" />
            </div>
          </div>

          <div className="w-full">
            <label className="block text-sm font-medium mb-2 text-gray-300">
              使用者 B
            </label>
            <input
              type="text"
              value={user2}
              onChange={(e) => setUser2(e.target.value)}
              placeholder="例如: TrashTaste"
              className="w-full px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-lg"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 rounded-lg font-bold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
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

      {error && (
        <div className="flex items-center justify-center gap-2 text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {result && (
        <div className="animate-fade-in space-y-8">
          {/* Profiles & Score */}
          <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 text-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-blue-500 to-teal-500" />

            <div className="flex flex-col md:flex-row items-center justify-center gap-8 mb-8">
              <a
                href={`https://anilist.co/user/${result.user1.name}`}
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

              <a
                href={`https://anilist.co/user/${result.user2.name}`}
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

          {/* Statistics Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <div className="flex items-center gap-3 mb-6">
                <a
                  href={`https://anilist.co/user/${result.user1.name}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-12 h-12 rounded-full border-4 border-blue-500 overflow-hidden hover:ring-2 hover:ring-blue-400 transition-all"
                >
                  <img
                    src={result.user1.avatar.large}
                    alt={result.user1.name}
                  />
                </a>
                <a
                  href={`https://anilist.co/user/${result.user1.name}`}
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

            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <div className="flex items-center gap-3 mb-6">
                <a
                  href={`https://anilist.co/user/${result.user2.name}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-12 h-12 rounded-full border-4 border-teal-500 overflow-hidden hover:ring-2 hover:ring-teal-400 transition-all"
                >
                  <img
                    src={result.user2.avatar.large}
                    alt={result.user2.name}
                  />
                </a>
                <a
                  href={`https://anilist.co/user/${result.user2.name}`}
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

          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-6 border border-purple-700">
              <div className="text-3xl font-bold mb-2">
                {result.common_count}
              </div>
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

          {/* Tabs Navigation */}
          <div className="flex gap-2 overflow-x-auto pb-2">
            <TabButton
              active={activeTab === "overview"}
              onClick={() => setActiveTab("overview")}
              icon={<BarChart3 className="w-5 h-5" />}
              label="品味分析"
            />
            <TabButton
              active={activeTab === "anime"}
              onClick={() => setActiveTab("anime")}
              icon={<Film className="w-5 h-5" />}
              label={`共同作品 (${result.common_count})`}
            />
            <TabButton
              active={activeTab === "recommendations"}
              onClick={() => setActiveTab("recommendations")}
              icon={<Heart className="w-5 h-5" />}
              label="互相推薦"
            />
            <TabButton
              active={activeTab === "disagreements"}
              onClick={() => setActiveTab("disagreements")}
              icon={<AlertCircle className="w-5 h-5" />}
              label="品味分歧"
            />
          </div>

          {/* Tab Content */}
          {activeTab === "overview" && (
            <div className="space-y-8">
              {/* Radar Chart */}
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
                <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-purple-400" />
                  類型偏好雷達圖
                </h3>
                <div className="w-full h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={formatRadarData(result.radar_data)}>
                      <PolarGrid stroke="#374151" />
                      <PolarAngleAxis
                        dataKey="subject"
                        tick={{ fill: "#9CA3AF" }}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        tick={{ fill: "#9CA3AF" }}
                      />
                      <Radar
                        name={result.user1.name}
                        dataKey={result.user1.name}
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.5}
                      />
                      <Radar
                        name={result.user2.name}
                        dataKey={result.user2.name}
                        stroke="#14B8A6"
                        fill="#14B8A6"
                        fillOpacity={0.5}
                      />
                      <Legend />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1F2937",
                          border: "1px solid #374151",
                          borderRadius: "0.5rem",
                        }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Common Genres */}
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
                          <span className="font-bold text-lg">
                            {genre.genre}
                          </span>
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
          )}

          {activeTab === "anime" && (
            <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Film className="w-6 h-6 text-blue-400" />
                共同觀看的動畫 ({result.common_count})
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {result.common_anime.map((anime) => (
                  <a
                    key={anime.id}
                    href={`https://anilist.co/anime/${anime.id}`}
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
                          <div
                            className="w-3 h-3 rounded-full border-2"
                            style={{ borderColor: "#3B82F6" }}
                          />
                          <span className="font-bold">{anime.user1_score}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div
                            className="w-3 h-3 rounded-full border-2"
                            style={{ borderColor: "#14B8A6" }}
                          />
                          <span className="font-bold">{anime.user2_score}</span>
                        </div>
                      </div>
                      {anime.score_diff > 0 && (
                        <div className="mt-2 text-xs text-gray-400 text-center">
                          差異: {anime.score_diff.toFixed(2)}
                        </div>
                      )}
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}

          {activeTab === "recommendations" && (
            <div className="space-y-8">
              {/* For User 2 */}
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
                <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Heart className="w-6 h-6 text-blue-400" />
                  推薦給 {result.user2.name}
                  <span className="text-sm text-gray-400 font-normal ml-2">
                    ({result.user1.name} 的高分作品)
                  </span>
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {result.recommendations.for_user2.length > 0 ? (
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

              {/* For User 1 */}
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
                <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Heart className="w-6 h-6 text-teal-400" />
                  推薦給 {result.user1.name}
                  <span className="text-sm text-gray-400 font-normal ml-2">
                    ({result.user2.name} 的高分作品)
                  </span>
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {result.recommendations.for_user1.length > 0 ? (
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
          )}

          {activeTab === "disagreements" && (
            <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold flex items-center gap-2">
                  <AlertCircle className="w-6 h-6 text-orange-400" />
                  品味分歧最大的作品
                </h3>
                {result.disagreements.length > 5 && (
                  <button
                    onClick={() => setShowDisagreementsModal(true)}
                    className="px-4 py-2 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-lg font-semibold transition-all flex items-center gap-2 shadow-lg"
                  >
                    查看全部 ({result.disagreements.length})
                    <AlertCircle className="w-4 h-4" />
                  </button>
                )}
              </div>
              <div className="space-y-4">
                {result.disagreements.length > 0 ? (
                  result.disagreements.slice(0, 5).map((anime, idx) => (
                    <div
                      key={anime.id}
                      className="bg-gray-700/50 rounded-lg p-4 flex flex-col md:flex-row gap-4 items-start md:items-center"
                    >
                      <div className="text-2xl font-bold text-gray-500 w-8">
                        #{idx + 1}
                      </div>
                      <a
                        href={`https://anilist.co/anime/${anime.id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <img
                          src={anime.coverImage}
                          alt={anime.title}
                          className="w-16 h-24 object-cover rounded hover:ring-2 hover:ring-orange-400 transition-all"
                        />
                      </a>
                      <div className="flex-1">
                        <a
                          href={`https://anilist.co/anime/${anime.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-bold text-lg mb-2 hover:text-blue-400 transition-colors block"
                        >
                          {anime.title}
                        </a>
                        <div className="flex items-center gap-6">
                          <div className="flex items-center gap-2">
                            <a
                              href={`https://anilist.co/user/${result.user1.name}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="w-10 h-10 rounded-full border-4 border-blue-500 overflow-hidden hover:ring-2 hover:ring-blue-400 transition-all"
                            >
                              <img
                                src={result.user1.avatar.large}
                                alt={result.user1.name}
                              />
                            </a>
                            <div>
                              <div className="text-xs text-gray-400">
                                {result.user1.name}
                              </div>
                              <div className="text-xl font-bold text-blue-400">
                                {anime.user1_score > 0
                                  ? anime.user1_score.toFixed(1)
                                  : "-"}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <a
                              href={`https://anilist.co/user/${result.user2.name}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="w-10 h-10 rounded-full border-4 border-teal-500 overflow-hidden hover:ring-2 hover:ring-teal-400 transition-all"
                            >
                              <img
                                src={result.user2.avatar.large}
                                alt={result.user2.name}
                              />
                            </a>
                            <div>
                              <div className="text-xs text-gray-400">
                                {result.user2.name}
                              </div>
                              <div className="text-xl font-bold text-teal-400">
                                {anime.user2_score > 0
                                  ? anime.user2_score.toFixed(1)
                                  : "-"}
                              </div>
                            </div>
                          </div>
                          <div className="ml-auto">
                            <div className="text-xs text-gray-400">
                              評分差異
                            </div>
                            <div className="text-2xl font-bold text-orange-400">
                              {anime.score_diff.toFixed(2)}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    <p>雙方評分非常一致，沒有明顯的品味分歧！</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Disagreements Modal */}
      {showDisagreementsModal && result && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in"
          onClick={() => setShowDisagreementsModal(false)}
        >
          <div
            className="bg-gray-800 rounded-2xl p-8 border border-orange-500/30 shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-700">
              <h3 className="text-2xl font-bold flex items-center gap-2">
                <AlertCircle className="w-6 h-6 text-orange-400" />
                所有品味分歧作品 ({result.disagreements.length})
              </h3>
              <button
                onClick={() => setShowDisagreementsModal(false)}
                className="text-gray-400 hover:text-white hover:bg-gray-700 transition-all text-2xl font-bold w-10 h-10 flex items-center justify-center rounded-lg"
                title="關閉"
              >
                ×
              </button>
            </div>
            <div className="space-y-4 overflow-y-auto pr-2 flex-1">
              {result.disagreements.length > 0 ? (
                result.disagreements.map((anime, idx) => (
                  <div
                    key={anime.id}
                    className="bg-gray-700/50 rounded-lg p-4 flex flex-col md:flex-row gap-4 items-start md:items-center hover:bg-gray-700/70 transition-colors"
                  >
                    <div className="text-2xl font-bold text-gray-500 w-8">
                      #{idx + 1}
                    </div>
                    <a
                      href={`https://anilist.co/anime/${anime.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <img
                        src={anime.coverImage}
                        alt={anime.title}
                        className="w-16 h-24 object-cover rounded hover:ring-2 hover:ring-orange-400 transition-all"
                      />
                    </a>
                    <div className="flex-1">
                      <a
                        href={`https://anilist.co/anime/${anime.id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-bold text-lg mb-2 hover:text-blue-400 transition-colors block"
                      >
                        {anime.title}
                      </a>
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                          <a
                            href={`https://anilist.co/user/${result.user1.name}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="w-10 h-10 rounded-full border-4 border-blue-500 overflow-hidden hover:ring-2 hover:ring-blue-400 transition-all"
                          >
                            <img
                              src={result.user1.avatar.large}
                              alt={result.user1.name}
                            />
                          </a>
                          <div>
                            <div className="text-xs text-gray-400">
                              {result.user1.name}
                            </div>
                            <div className="text-xl font-bold text-blue-400">
                              {anime.user1_score > 0
                                ? anime.user1_score.toFixed(1)
                                : "-"}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <a
                            href={`https://anilist.co/user/${result.user2.name}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="w-10 h-10 rounded-full border-4 border-teal-500 overflow-hidden hover:ring-2 hover:ring-teal-400 transition-all"
                          >
                            <img
                              src={result.user2.avatar.large}
                              alt={result.user2.name}
                            />
                          </a>
                          <div>
                            <div className="text-xs text-gray-400">
                              {result.user2.name}
                            </div>
                            <div className="text-xl font-bold text-teal-400">
                              {anime.user2_score > 0
                                ? anime.user2_score.toFixed(1)
                                : "-"}
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
                ))
              ) : (
                <div className="text-center py-12 text-gray-400">
                  <AlertCircle className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg">沒有找到評分差異的作品</p>
                  <p className="text-sm mt-2">可能雙方都沒有對共同作品評分</p>
                </div>
              )}
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700 text-center text-sm text-gray-400">
              使用滑鼠滾輪或觸控滑動瀏覽 • 點擊作品查看詳情
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper Components
const StatItem = ({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number | string;
}) => (
  <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
    <div className="flex items-center gap-3">
      <div className="text-gray-400">{icon}</div>
      <span className="text-gray-300">{label}</span>
    </div>
    <span className="text-xl font-bold">{value}</span>
  </div>
);

const TabButton = ({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap ${
      active
        ? "bg-gradient-to-r from-blue-600 to-teal-600 text-white"
        : "bg-gray-700 text-gray-300 hover:bg-gray-600"
    }`}
  >
    {icon}
    {label}
  </button>
);

const RecommendationCard = ({ anime }: { anime: Recommendation }) => (
  <a
    href={`https://anilist.co/anime/${anime.id}`}
    target="_blank"
    rel="noopener noreferrer"
    className="bg-gray-700/50 rounded-lg overflow-hidden hover:ring-2 hover:ring-purple-500 transition-all cursor-pointer group block"
  >
    <div className="relative">
      <img
        src={anime.coverImage}
        alt={anime.title}
        className="w-full h-48 object-cover group-hover:scale-105 transition-transform"
      />
      <div className="absolute top-2 right-2 bg-yellow-500 text-black px-2 py-1 rounded-full text-xs font-bold flex items-center gap-1">
        <Star className="w-3 h-3 fill-black" />
        {anime.score.toFixed(1)}
      </div>
      {!anime.user_scored && (
        <div className="absolute top-2 left-2 bg-gray-800/90 text-white px-2 py-1 rounded text-xs">
          社群評分
        </div>
      )}
    </div>
    <div className="p-3">
      <h4 className="font-semibold text-sm line-clamp-2 h-10 mb-2">
        {anime.title}
      </h4>
      <div className="flex flex-wrap gap-1">
        {anime.genres.slice(0, 2).map((genre) => (
          <span key={genre} className="text-xs bg-gray-600 px-2 py-1 rounded">
            {genre}
          </span>
        ))}
      </div>
    </div>
  </a>
);
