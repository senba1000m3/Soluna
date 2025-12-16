import React from "react";
import { Info } from "lucide-react";

export interface AnimeItem {
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

interface AnimeCardProps {
  anime: AnimeItem;
  variant?: "watching" | "planning" | "dropped";
  onShowDetails?: (anime: AnimeItem) => void;
}

export const AnimeCard: React.FC<AnimeCardProps> = ({
  anime,
  variant = "watching",
  onShowDetails,
}) => {
  const getBorderColor = () => {
    if (variant === "dropped") return "hover:border-red-500/50";
    if (variant === "planning") return "hover:border-purple-500/50";
    return "hover:border-yellow-500/50";
  };

  const getTitleHoverColor = () => {
    if (variant === "dropped") return "group-hover:text-red-400";
    if (variant === "planning") return "group-hover:text-purple-400";
    return "group-hover:text-yellow-400";
  };

  const getRiskColor = (probability: number) => {
    if (probability > 0.7) return "text-red-500";
    if (probability > 0.4) return "text-yellow-500";
    return "text-green-500";
  };

  const getBarColor = (probability: number) => {
    if (probability > 0.7) return "bg-red-500";
    if (probability > 0.4) return "bg-yellow-500";
    return "bg-green-500";
  };

  return (
    <div
      className={`bg-gray-800 rounded-xl overflow-hidden border border-gray-700 ${getBorderColor()} transition-all shadow-lg group`}
    >
      <div className="flex h-36">
        <div className="w-24 flex-shrink-0">
          <img
            src={anime.cover}
            alt={anime.title}
            className="w-full h-full object-cover"
          />
        </div>
        <div className="p-4 flex-1 flex flex-col justify-between">
          <div className="space-y-2">
            <h3
              className={`font-bold text-white line-clamp-2 ${getTitleHoverColor()} transition-colors`}
            >
              {anime.title}
            </h3>

            {/* Show drop probability for watching/planning */}
            {variant !== "dropped" && anime.drop_probability !== undefined && (
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-gray-400">棄番風險</span>
                  <span
                    className={`font-bold ${getRiskColor(anime.drop_probability)}`}
                  >
                    {(anime.drop_probability * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-2">
                  <div
                    className={`h-full ${getBarColor(anime.drop_probability)}`}
                    style={{
                      width: `${anime.drop_probability * 100}%`,
                    }}
                  />
                </div>
                {anime.drop_reasons && anime.drop_reasons.length > 0 && (
                  <button
                    onClick={() => onShowDetails?.(anime)}
                    className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 transition-colors"
                  >
                    <Info className="w-3 h-3" />
                    查看判斷依據
                  </button>
                )}
              </div>
            )}

            {/* Show genres for dropped anime */}
            {variant === "dropped" && (
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
            )}
          </div>

          {/* Progress and score info */}
          {variant === "dropped" && (
            <div className="flex items-center justify-between text-xs text-gray-400 mt-2">
              <span>
                進度: {anime.progress} / {anime.total_episodes || "?"}
              </span>
              {anime.score > 0 && (
                <span className="text-yellow-500 font-bold">
                  {anime.score}分
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Progress Bar for dropped anime */}
      {variant === "dropped" && (
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
      )}
    </div>
  );
};
