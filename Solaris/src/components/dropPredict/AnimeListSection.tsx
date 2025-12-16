import React from "react";
import { AlertTriangle, XCircle, CheckCircle } from "lucide-react";
import { AnimeCard, AnimeItem } from "./AnimeCard";

interface AnimeListSectionProps {
  title: string;
  animeList: AnimeItem[];
  variant: "watching" | "planning" | "dropped";
  icon?: "warning" | "x" | "check";
  iconColor?: string;
  emptyMessage?: string;
  emptySubMessage?: string;
  onShowDetails?: (anime: AnimeItem) => void;
  limit?: number;
}

export const AnimeListSection: React.FC<AnimeListSectionProps> = ({
  title,
  animeList,
  variant,
  icon = "warning",
  iconColor = "text-yellow-500",
  emptyMessage,
  emptySubMessage,
  onShowDetails,
  limit,
}) => {
  const getIcon = () => {
    switch (icon) {
      case "x":
        return <XCircle className={`w-6 h-6 ${iconColor}`} />;
      case "check":
        return <CheckCircle className={`w-6 h-6 ${iconColor}`} />;
      default:
        return <AlertTriangle className={`w-6 h-6 ${iconColor}`} />;
    }
  };

  // Filter anime with drop probability for watching/planning
  const filteredList =
    variant === "dropped"
      ? animeList
      : animeList.filter((a) => (a.drop_probability ?? 0) > 0);

  // Apply limit if specified
  const displayList = limit ? filteredList.slice(0, limit) : filteredList;

  // Don't render if list is empty
  if (filteredList.length === 0) {
    // Show empty state for dropped list
    if (variant === "dropped" && emptyMessage) {
      return (
        <div>
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            {getIcon()}
            {title}
          </h2>
          <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-white mb-2">
              {emptyMessage}
            </h3>
            {emptySubMessage && (
              <p className="text-gray-400">{emptySubMessage}</p>
            )}
          </div>
        </div>
      );
    }
    return null;
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
        {getIcon()}
        {title}
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {displayList.map((anime) => (
          <AnimeCard
            key={anime.id}
            anime={anime}
            variant={variant}
            onShowDetails={onShowDetails}
          />
        ))}
      </div>
    </div>
  );
};
