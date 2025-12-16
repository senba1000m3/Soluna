import React, { useState } from "react";
import { Sparkles, Loader2, ExternalLink } from "lucide-react";
import { QuickIDSelector } from "../components/QuickIDSelector";
import {
  AnimeCard,
  MatchReasonModal,
  SeasonSelector,
} from "../components/recommend";

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
        <div className="space-y-4">
          <QuickIDSelector
            value={username}
            onChange={setUsername}
            label="AniList 使用者名稱（選填）"
            placeholder="例如：senba1000m3"
            required={false}
          />

          <SeasonSelector
            year={year}
            season={season}
            onYearChange={setYear}
            onSeasonChange={setSeason}
            nextSeason={nextSeason}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full mt-5 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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
          <AnimeCard
            key={anime.id}
            id={anime.id}
            title={anime.title}
            coverImage={anime.coverImage}
            genres={anime.genres}
            averageScore={anime.averageScore}
            popularity={anime.popularity}
            matchScore={anime.match_score}
            matchReasons={anime.match_reasons}
            onInfoClick={() => setSelectedAnime(anime)}
          />
        ))}
      </div>

      {!loading && results.length === 0 && !error && (
        <p className="text-center text-gray-500 mt-8">
          請輸入條件後點擊「取得推薦」
        </p>
      )}

      {/* Modal for match reasons */}
      {selectedAnime && selectedAnime.match_reasons && (
        <MatchReasonModal
          isOpen={true}
          onClose={() => setSelectedAnime(null)}
          animeTitle={selectedAnime.title}
          matchScore={selectedAnime.match_score!}
          matchReasons={selectedAnime.match_reasons}
        />
      )}
    </div>
  );
};
