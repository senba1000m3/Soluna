import React, { useState } from "react";
import {
  Sparkles,
  Loader2,
  Star,
  Calendar,
  Info,
  X,
  ExternalLink,
} from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface AnimeTitle {
  romaji: string;
  english: string | null;
}

interface AnimeCoverImage {
  large: string;
}

interface MatchReason {
  matched_genres: Array<{ genre: string; weight: number }>;
  total_weight: number;
  top_reason: string;
}

interface Anime {
  id: number;
  title: AnimeTitle;
  description: string;
  genres: string[];
  averageScore: number;
  popularity: number;
  coverImage: AnimeCoverImage;
  match_score?: number;
  match_reasons?: MatchReason;
}

interface RecommendResponse {
  season: string;
  year: number;
  display_season: string;
  recommendations: Anime[];
}

export const Recommend = () => {
  // Calculate next season
  const getNextSeason = () => {
    const now = new Date();
    const month = now.getMonth() + 1; // 1-12
    const year = now.getFullYear();

    if (month >= 1 && month <= 3) {
      return { season: "SPRING", year, label: `春-4 月 (${year})` };
    } else if (month >= 4 && month <= 6) {
      return { season: "SUMMER", year, label: `夏-7 月 (${year})` };
    } else if (month >= 7 && month <= 9) {
      return { season: "FALL", year, label: `秋-10 月 (${year})` };
    } else {
      return {
        season: "WINTER",
        year: year + 1,
        label: `冬-1 月 (${year + 1})`,
      };
    }
  };

  const nextSeason = getNextSeason();

  const [username, setUsername] = useState("");
  const [year, setYear] = useState<string>(nextSeason.year.toString());
  const [season, setSeason] = useState<string>(nextSeason.season);
  const [results, setResults] = useState<Anime[]>([]);
  const [displayInfo, setDisplayInfo] = useState<{
    season: string;
    year: number;
    display_season: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedAnime, setSelectedAnime] = useState<Anime | null>(null);

  const seasonOptions = [
    { value: "WINTER", label: "冬-1 月" },
    { value: "SPRING", label: "春-4 月" },
    { value: "SUMMER", label: "夏-7 月" },
    { value: "FALL", label: "秋-10 月" },
  ];

  const handleRecommend = async (e: React.FormEvent) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResults([]);
    setDisplayInfo(null);

    try {
      const payload: any = {};
      if (username.trim()) payload.username = username.trim();
      if (year) payload.year = parseInt(year);
      if (season) payload.season = season;

      const response = await fetch(`${BACKEND_URL}/recommend`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("推薦請求失敗");
      }

      const data: RecommendResponse = await response.json();
      setResults(data.recommendations);
      setDisplayInfo({
        season: data.season,
        year: data.year,
        display_season: data.display_season,
      });
    } catch (err) {
      setError("發生錯誤，請稍後再試");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
          新番推薦系統
        </h1>
        <p className="text-gray-400">基於你的喜好，推薦最適合的當季新番</p>
      </div>

      <form
        onSubmit={handleRecommend}
        className="mb-12 bg-gray-800 p-6 rounded-xl border border-gray-700"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              AniList 使用者名稱 (選填)
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="例如: Gigguk"
              className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
            />
            <p className="text-xs text-gray-500 mt-1">留空則顯示熱門排序</p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              年份
              <span className="ml-2 text-xs text-purple-400">
                (下季: {nextSeason.year}{" "}
                {
                  seasonOptions
                    .find((o) => o.value === nextSeason.season)
                    ?.label?.split("-")[0]
                }
                )
              </span>
            </label>
            <input
              type="number"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              placeholder="例如: 2025"
              className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              季度
            </label>
            <select
              value={season}
              onChange={(e) => setSeason(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
            >
              {seasonOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              分析中...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              取得推薦
            </>
          )}
        </button>
      </form>

      {error && (
        <div className="text-center text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800">
          {error}
        </div>
      )}

      {displayInfo && (
        <div className="text-center mb-6 text-gray-300">
          <p className="text-lg flex items-center justify-center gap-2">
            <a
              href={`https://anilist.co/search/anime?year=${displayInfo.year}&season=${displayInfo.season}`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-semibold text-purple-400 hover:text-purple-300 hover:underline flex items-center gap-1 transition-colors"
            >
              {displayInfo.display_season} ({displayInfo.year})
              <ExternalLink className="w-4 h-4" />
            </a>
            推薦結果
            {username && (
              <span className="text-sm text-gray-500">
                {" "}
                - 為 {username} 個人化推薦
              </span>
            )}
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((anime) => (
          <div
            key={anime.id}
            className="bg-gray-800 rounded-xl overflow-hidden hover:transform hover:scale-[1.02] transition-all duration-300 shadow-lg border border-gray-700/50"
          >
            <div className="relative">
              <img
                src={anime.coverImage.large}
                alt={anime.title.romaji}
                className="w-full h-64 object-cover"
              />
              {anime.match_score !== undefined && anime.match_reasons && (
                <>
                  <button
                    onClick={() => setSelectedAnime(anime)}
                    className="absolute top-2 left-2 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-full shadow-lg transition-colors"
                    title="查看匹配原因"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                  <div className="absolute top-2 right-2 bg-purple-600 text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg">
                    {anime.match_score.toFixed(0)}% 匹配
                  </div>
                </>
              )}
            </div>
            <div className="p-4">
              <h3 className="font-bold text-lg line-clamp-2 mb-2 text-purple-100">
                {anime.title.english || anime.title.romaji}
              </h3>
              <div className="flex flex-wrap gap-2 mb-3">
                {anime.genres.slice(0, 3).map((genre) => (
                  <span
                    key={genre}
                    className="text-xs px-2 py-1 bg-gray-700 rounded-full text-gray-300"
                  >
                    {genre}
                  </span>
                ))}
              </div>
              <div className="flex items-center justify-between text-sm text-gray-400">
                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                  <span>{anime.averageScore || "N/A"}%</span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  <span>{anime.popularity?.toLocaleString() || "N/A"}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {!loading && results.length === 0 && !error && (
        <p className="text-center text-gray-500 mt-8">
          請輸入條件後點擊「取得推薦」
        </p>
      )}

      {/* Modal for match reasons */}
      {selectedAnime && selectedAnime.match_reasons && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedAnime(null)}
        >
          <div
            className="bg-gray-800 rounded-xl max-w-md w-full p-6 border border-gray-700 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-xl font-bold text-purple-300">匹配原因</h2>
              <button
                onClick={() => setSelectedAnime(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="mb-4">
              <h3 className="font-semibold text-lg text-white mb-2">
                {selectedAnime.title.english || selectedAnime.title.romaji}
              </h3>
              <div className="flex items-center gap-2 mb-3">
                <div className="bg-purple-600 px-3 py-1 rounded-full text-sm font-bold">
                  {selectedAnime.match_score?.toFixed(0)}% 匹配度
                </div>
              </div>
            </div>

            <div className="bg-gray-900 p-4 rounded-lg mb-4">
              <p className="text-green-400 font-medium mb-3">
                ✨ {selectedAnime.match_reasons.top_reason}
              </p>

              {selectedAnime.match_reasons.matched_genres.length > 0 && (
                <div>
                  <p className="text-gray-400 text-sm mb-2">你喜歡的類型：</p>
                  <div className="space-y-2">
                    {selectedAnime.match_reasons.matched_genres.map(
                      (genre, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between"
                        >
                          <span className="text-purple-300 font-medium">
                            {genre.genre}
                          </span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-gray-700 rounded-full h-2 overflow-hidden">
                              <div
                                className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full"
                                style={{
                                  width: `${Math.min((genre.weight / selectedAnime.match_reasons!.total_weight) * 100, 100)}%`,
                                }}
                              />
                            </div>
                            <span className="text-xs text-gray-500 w-12 text-right">
                              {(
                                (genre.weight /
                                  selectedAnime.match_reasons!.total_weight) *
                                100
                              ).toFixed(0)}
                              %
                            </span>
                          </div>
                        </div>
                      ),
                    )}
                  </div>
                </div>
              )}

              {selectedAnime.match_reasons.matched_genres.length === 0 && (
                <p className="text-gray-400 text-sm">
                  這部作品符合你的整體觀看偏好，建議試試看！
                </p>
              )}
            </div>

            <div className="text-xs text-gray-500 bg-gray-900 p-3 rounded">
              <p className="mb-1">
                📊 <strong>計算方式：</strong>
              </p>
              <p>
                根據你在 AniList
                上的評分記錄，分析你喜愛的類型權重，與當季新番進行餘弦相似度匹配。
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
