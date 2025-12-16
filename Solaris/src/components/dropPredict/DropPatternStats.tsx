import React from "react";

export interface DropPatternStat {
  name: string;
  dropped: number;
  completed: number;
  total: number;
  drop_rate: number;
}

export interface DropPatterns {
  top_dropped_tags: DropPatternStat[];
  top_dropped_genres: DropPatternStat[];
  top_dropped_studios: DropPatternStat[];
}

interface DropPatternStatsProps {
  patterns: DropPatterns;
}

const PatternCard: React.FC<{
  title: string;
  emoji: string;
  stats: DropPatternStat[];
}> = ({ title, emoji, stats }) => {
  if (stats.length === 0) return null;

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h3 className="text-xl font-bold text-white mb-4">
        {emoji} {title}
      </h3>
      <div className="space-y-3">
        {stats.slice(0, 5).map((stat, idx) => (
          <div key={idx} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-300 truncate">{stat.name}</span>
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
  );
};

export const DropPatternStats: React.FC<DropPatternStatsProps> = ({
  patterns,
}) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <PatternCard
        title="最容易棄番的標籤"
        emoji="🏷️"
        stats={patterns.top_dropped_tags}
      />
      <PatternCard
        title="最容易棄番的類型"
        emoji="🎭"
        stats={patterns.top_dropped_genres}
      />
      <PatternCard
        title="最容易棄番的製作公司"
        emoji="🏢"
        stats={patterns.top_dropped_studios}
      />
    </div>
  );
};
