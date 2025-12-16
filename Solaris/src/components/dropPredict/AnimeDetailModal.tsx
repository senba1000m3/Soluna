import React from "react";
import { X, Info } from "lucide-react";
import { AnimeItem } from "./AnimeCard";

interface AnimeDetailModalProps {
  anime: AnimeItem | null;
  onClose: () => void;
}

export const AnimeDetailModal: React.FC<AnimeDetailModalProps> = ({
  anime,
  onClose,
}) => {
  if (!anime) return null;

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
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-gray-600 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-gray-800 border-b border-gray-700 p-6 flex items-start gap-4">
          <img
            src={anime.cover}
            alt={anime.title}
            className="w-20 h-28 object-cover rounded-lg"
          />
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white mb-2">
              {anime.title}
            </h2>
            {anime.drop_probability !== undefined && (
              <div className="flex items-center gap-2">
                <span className="text-gray-400 text-sm">棄番風險:</span>
                <span
                  className={`text-2xl font-bold ${getRiskColor(anime.drop_probability)}`}
                >
                  {(anime.drop_probability * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Risk Bar */}
          {anime.drop_probability !== undefined && (
            <div>
              <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full ${getBarColor(anime.drop_probability)}`}
                  style={{
                    width: `${anime.drop_probability * 100}%`,
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>低風險</span>
                <span>高風險</span>
              </div>
            </div>
          )}

          {/* Reasons */}
          {anime.drop_reasons && anime.drop_reasons.length > 0 && (
            <div>
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Info className="w-5 h-5 text-blue-400" />
                判斷依據
              </h3>
              <div className="space-y-3">
                {anime.drop_reasons.map((reason, idx) => (
                  <div
                    key={idx}
                    className="bg-gray-900/50 p-4 rounded-lg border border-gray-700"
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-blue-400 font-bold text-lg mt-0.5">
                        {idx + 1}.
                      </span>
                      <span className="text-gray-300 flex-1">{reason}</span>
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
                  {anime.genres.join(", ")}
                </span>
              </div>
              <div className="flex">
                <span className="text-gray-400 w-24">進度:</span>
                <span className="text-gray-300">
                  {anime.progress} / {anime.total_episodes || "?"}
                </span>
              </div>
              {anime.score > 0 && (
                <div className="flex">
                  <span className="text-gray-400 w-24">評分:</span>
                  <span className="text-yellow-500 font-bold">
                    {anime.score} 分
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
