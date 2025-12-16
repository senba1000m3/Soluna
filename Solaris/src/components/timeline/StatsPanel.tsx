import React from "react";
import { Trophy } from "lucide-react";
import { StatCard } from "./StatCard";

interface TimelineStats {
  most_active_year?: {
    year: number;
    count: number;
    label: string;
  };
  favorite_genre?: {
    genre: string;
    count: number;
    label: string;
  };
  total_watch_time?: {
    days: number;
    hours: number;
    label: string;
  };
  favorite_season?: {
    season: string;
    season_en: string;
    count: number;
    year: number;
    label: string;
  };
  favorites_count?: {
    count: number;
    label: string;
    description: string;
  };
}

interface StatsPanelProps {
  stats: TimelineStats;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({ stats }) => {
  if (!stats || Object.keys(stats).length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2 border-b border-gray-700 pb-3">
        <Trophy className="w-5 h-5 text-amber-500" />
        你的動畫數據
      </h3>
      <div className="space-y-6">
        {stats.most_active_year && (
          <StatCard
            label={stats.most_active_year.label}
            value={stats.most_active_year.year}
            subtitle={`看了 ${stats.most_active_year.count} 部`}
          />
        )}
        {stats.favorite_genre && (
          <StatCard
            label={stats.favorite_genre.label}
            value={stats.favorite_genre.genre}
            subtitle={`看了 ${stats.favorite_genre.count} 部`}
          />
        )}
        {stats.total_watch_time && (
          <StatCard
            label={stats.total_watch_time.label}
            value={`${stats.total_watch_time.days} 天`}
            subtitle={`約 ${stats.total_watch_time.hours} 小時`}
          />
        )}
        {stats.favorite_season && (
          <StatCard
            label={stats.favorite_season.label}
            value={`${stats.favorite_season.year} ${stats.favorite_season.season}`}
            subtitle={`追了 ${stats.favorite_season.count} 部`}
          />
        )}
        {stats.favorites_count && (
          <StatCard
            label={stats.favorites_count.label}
            value={stats.favorites_count.count}
            subtitle={stats.favorites_count.description}
          />
        )}
      </div>
    </div>
  );
};
