import React from "react";
import { Star } from "lucide-react";
import { Recommendation } from "../../types/synergy";
import { getAnimeUrl, formatScore } from "../../utils/synergyHelpers";

interface RecommendationCardProps {
  anime: Recommendation;
}

/**
 * 推薦卡片組件
 * 用於顯示互相推薦的動畫作品
 * 包含封面、評分、類型標籤等信息
 */
export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  anime,
}) => {
  return (
    <a
      href={getAnimeUrl(anime.id)}
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
        {/* 評分徽章 */}
        <div className="absolute top-2 right-2 bg-yellow-500 text-black px-2 py-1 rounded-full text-xs font-bold flex items-center gap-1">
          <Star className="w-3 h-3 fill-black" />
          {formatScore(anime.score, 1)}
        </div>
        {/* 社群評分標記 */}
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
        {/* 類型標籤 */}
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
};
