import React, { useState } from "react";
import { Search, Loader2, ExternalLink, Star, Calendar } from "lucide-react";
import { BACKEND_URL } from "../config/env";

// Define types based on the backend response
interface AnimeTitle {
  romaji: string;
  english: string | null;
}

interface AnimeCoverImage {
  large: string;
}

interface Anime {
  id: number;
  title: AnimeTitle;
  description: string;
  genres: string[];
  averageScore: number;
  status: string;
  season: string;
  seasonYear: number;
  coverImage: AnimeCoverImage;
}

interface SearchResponse {
  results: Anime[];
}

export const Soluna = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Anime[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error("Search failed:", err);
      setError("搜尋失敗，請確認後端服務正在運行或稍後再試。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-5 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 text-transparent bg-clip-text">
            Soluna 動畫搜尋
          </h1>
          <p className="text-gray-400 text-lg">探索你最愛的動畫作品</p>
        </header>

        {/* Search Form */}
        <form onSubmit={handleSearch} className="max-w-3xl mx-auto mb-12">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="輸入動畫名稱，例如：刀劍神域、遊戲人生..."
              className="w-full pl-12 pr-32 py-4 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  搜尋中...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  搜尋
                </>
              )}
            </button>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="max-w-3xl mx-auto mb-8 bg-red-900/20 border border-red-800 rounded-xl p-4 text-center">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <Loader2 className="w-12 h-12 animate-spin text-purple-400 mx-auto mb-4" />
            <p className="text-gray-400">正在搜尋...</p>
          </div>
        )}

        {/* Results Grid */}
        {!loading && results.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-12">
            {results.map((anime) => (
              <a
                key={anime.id}
                href={`https://anilist.co/anime/${anime.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="group bg-gray-800 rounded-xl overflow-hidden border border-gray-700 hover:border-purple-500 transition-all duration-300 hover:transform hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/20"
              >
                {/* Cover Image */}
                <div className="relative overflow-hidden aspect-[9/10]">
                  <img
                    src={anime.coverImage.large}
                    alt={anime.title.romaji}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                  {/* Score Badge */}
                  {anime.averageScore && (
                    <div className="absolute top-3 right-3 bg-yellow-500/90 backdrop-blur-sm text-gray-900 px-2 py-1 rounded-lg flex items-center gap-1 font-bold text-sm">
                      <Star className="w-4 h-4 fill-current" />
                      {anime.averageScore}
                    </div>
                  )}

                  {/* Status Badge */}
                  <div className="absolute top-3 left-3 bg-gray-900/90 backdrop-blur-sm text-gray-200 px-2 py-1 rounded-lg text-xs font-medium">
                    {anime.status === "FINISHED" && "已完結"}
                    {anime.status === "RELEASING" && "連載中"}
                    {anime.status === "NOT_YET_RELEASED" && "未上映"}
                    {anime.status === "CANCELLED" && "已取消"}
                  </div>

                  {/* External Link Icon */}
                  <div className="absolute bottom-3 right-3 bg-purple-500/90 backdrop-blur-sm p-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
                    <ExternalLink className="w-4 h-4 text-white" />
                  </div>
                </div>

                {/* Card Content */}
                <div className="p-4">
                  {/* Title */}
                  <h3 className="font-bold text-white line-clamp-2 min-h-[3rem] group-hover:text-purple-400 transition-colors">
                    {anime.title.english || anime.title.romaji}
                  </h3>

                  {/* Season Info */}
                  <div className="flex items-center gap-1 text-gray-400 text-sm mb-3">
                    <Calendar className="w-4 h-4" />
                    <span>
                      {anime.season && `${anime.season} `}
                      {anime.seasonYear}
                    </span>
                  </div>

                  {/* Genres */}
                  <div className="flex flex-wrap gap-2">
                    {anime.genres.slice(0, 3).map((genre) => (
                      <span
                        key={genre}
                        className="px-2 py-1 bg-gray-700 text-gray-300 rounded-md text-xs font-medium"
                      >
                        {genre}
                      </span>
                    ))}
                    {anime.genres.length > 3 && (
                      <span className="px-2 py-1 bg-gray-700 text-gray-400 rounded-md text-xs">
                        +{anime.genres.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              </a>
            ))}
          </div>
        )}

        {/* No Results */}
        {!loading && results.length === 0 && query && !error && (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🔍</div>
            <p className="text-gray-400 text-lg">找不到相關結果</p>
            <p className="text-gray-500 text-sm mt-2">請嘗試其他關鍵字</p>
          </div>
        )}

        {/* Empty State */}
        {!loading && results.length === 0 && !query && !error && (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">✨</div>
            <p className="text-gray-400 text-lg">開始搜尋你喜歡的動畫作品</p>
            <p className="text-gray-500 text-sm mt-2">
              輸入動畫名稱並按下搜尋按鈕
            </p>
          </div>
        )}

        {/* Footer - AniList Attribution */}
        <footer className="mt-16 pt-8 border-t border-gray-800">
          <div className="text-center">
            <p className="text-gray-400 text-sm mb-2">數據來源</p>
            <a
              href="https://anilist.co"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors font-medium"
            >
              <svg
                className="w-5 h-5"
                viewBox="0 0 24 24"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M6.361 2.943L0 21.056h4.942l1.077-3.133H11.4l1.052 3.133H22.9c.71 0 1.1-.392 1.1-1.101V17.53c0-.71-.39-1.101-1.1-1.101h-6.483V4.045c0-.71-.392-1.102-1.101-1.102h-2.422c-.71 0-1.101.392-1.101 1.102v1.064l-.758-2.166zm2.324 5.948l1.688 5.018H7.144z" />
              </svg>
              AniList
              <ExternalLink className="w-4 h-4" />
            </a>
            <p className="text-gray-500 text-xs mt-3">
              本服務使用 AniList API 提供動畫資訊
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};
