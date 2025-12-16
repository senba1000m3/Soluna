import React from "react";

interface SeasonOption {
  value: string;
  label: string;
}

interface SeasonSelectorProps {
  year: string;
  season: string;
  onYearChange: (year: string) => void;
  onSeasonChange: (season: string) => void;
  nextSeason: {
    year: number;
    season: string;
    label: string;
  };
}

const seasonOptions: SeasonOption[] = [
  { value: "WINTER", label: "冬-1 月" },
  { value: "SPRING", label: "春-4 月" },
  { value: "SUMMER", label: "夏-7 月" },
  { value: "FALL", label: "秋-10 月" },
];

export const SeasonSelector: React.FC<SeasonSelectorProps> = ({
  year,
  season,
  onYearChange,
  onSeasonChange,
  nextSeason,
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium mb-2 text-gray-300">
          年份
          <span className="ml-2 text-xs text-purple-400">
            (下季: {nextSeason.year}{" "}
            {
              seasonOptions
                .find((o) => o.value === nextSeason.season)
                ?.label?.split("-")[0]
            }
            )
          </span>
        </label>
        <input
          type="number"
          value={year}
          onChange={(e) => onYearChange(e.target.value)}
          placeholder="例如：2025"
          className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2 text-gray-300">
          季度
        </label>
        <select
          value={season}
          onChange={(e) => onSeasonChange(e.target.value)}
          className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
        >
          {seasonOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};
