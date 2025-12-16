import React from "react";
import { Star, Calendar, Info } from "lucide-react";

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

interface AnimeCardProps {
  id: number;
  title: AnimeTitle;
  coverImage: AnimeCoverImage;
  genres: string[];
  averageScore: number;
  popularity: number;
  matchScore?: number;
  matchReasons?: MatchReason;
  onInfoClick?: () => void;
  onCardClick?: () => void;
}

export const AnimeCard: React.FC<AnimeCardProps> = ({
  id,
  title,
  coverImage,
  genres,
  averageScore,
  popularity,
  matchScore,
  matchReasons,
  onInfoClick,
  onCardClick,
}) => {
  const handleCardClick = () => {
    if (onCardClick) {
      onCardClick();
    } else {
      window.open(`https://anilist.co/anime/${id}`, "_blank");
    }
  };

  const handleInfoClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onInfoClick) {
      onInfoClick();
    }
  };

  return (
    <div
      className="bg-gray-800 rounded-xl overflow-hidden hover:transform hover:scale-[1.02] transition-all duration-300 shadow-lg border border-gray-700/50 cursor-pointer"
      onClick={handleCardClick}
    >
      <div className="relative">
        <img
          src={coverImage.large}
          alt={title.romaji}
          className="w-full h-64 object-cover"
        />
        {matchScore !== undefined && matchReasons && (
          <>
            <button
              onClick={handleInfoClick}
              className="absolute top-2 left-2 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-full shadow-lg transition-colors"
              title="查看匹配原因"
            >
              <Info className="w-4 h-4" />
            </button>
            <div className="absolute top-2 right-2 bg-purple-600 text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg">
              {matchScore.toFixed(0)}% 匹配
            </div>
          </>
        )}
      </div>
      <div className="p-4">
        <h3 className="font-bold text-lg line-clamp-2 mb-2 text-purple-100 hover:text-purple-300 transition-colors">
          {title.english || title.romaji}
        </h3>
        <div className="flex flex-wrap gap-2 mb-3">
          {genres.slice(0, 3).map((genre) => (
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
            <span>{averageScore || "N/A"}%</span>
          </div>
          <div className="flex items-center gap-1">
            <Calendar className="w-4 h-4" />
            <span>{popularity?.toLocaleString() || "N/A"}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
