import React from "react";
import { X } from "lucide-react";

interface AnimeTitle {
  romaji: string;
  english: string | null;
}

interface MatchReason {
  matched_genres: Array<{ genre: string; weight: number }>;
  total_weight: number;
  top_reason: string;
}

interface MatchReasonModalProps {
  isOpen: boolean;
  onClose: () => void;
  animeTitle: AnimeTitle;
  matchScore: number;
  matchReasons: MatchReason;
}

export const MatchReasonModal: React.FC<MatchReasonModalProps> = ({
  isOpen,
  onClose,
  animeTitle,
  matchScore,
  matchReasons,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-xl max-w-md w-full p-6 border border-gray-700 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-start mb-4">
          <h2 className="text-xl font-bold text-purple-300">匹配原因</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="mb-4">
          <h3 className="font-semibold text-lg text-white mb-2">
            {animeTitle.english || animeTitle.romaji}
          </h3>
          <div className="flex items-center gap-2 mb-3">
            <div className="bg-purple-600 px-3 py-1 rounded-full text-sm font-bold">
              匹配度：{matchScore.toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="bg-gray-900 p-4 rounded-lg mb-4">
          <p className="text-green-400 font-medium mb-3">
            ✨ {matchReasons.top_reason}
          </p>

          {matchReasons.matched_genres.length > 0 && (
            <div>
              <p className="text-gray-400 text-sm mb-2">你喜歡的類型：</p>
              <div className="space-y-2">
                {matchReasons.matched_genres.map((genre, idx) => (
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
                            width: `${Math.min((genre.weight / matchReasons.total_weight) * 100, 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 w-12 text-right">
                        {((genre.weight / matchReasons.total_weight) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {matchReasons.matched_genres.length === 0 && (
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
  );
};
