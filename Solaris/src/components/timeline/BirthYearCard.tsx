import React from "react";
import { Star } from "lucide-react";

interface BirthYearAnime {
  id: number;
  title: {
    romaji: string;
    english: string | null;
  };
  coverImage: {
    large: string;
  };
  startDate: {
    year: number;
    month: number;
    day: number;
  };
  averageScore: number;
  popularity: number;
  seasonYear: number;
  format: string;
}

interface BirthYearCardProps {
  birthYear: number;
  animeList: BirthYearAnime[];
}

export const BirthYearCard: React.FC<BirthYearCardProps> = ({ animeList }) => {
  if (!animeList || animeList.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2 border-b border-gray-700 pb-3">
        <Star className="w-5 h-5 text-amber-500" />
        你出生那年的霸權動畫
      </h3>
      <div className="grid grid-cols-5 gap-4">
        {animeList.slice(0, 10).map((anime) => (
          <div
            key={anime.id}
            className="text-center group cursor-pointer"
            onClick={() =>
              window.open(`https://anilist.co/anime/${anime.id}`, "_blank")
            }
          >
            <div className="relative mb-2 overflow-hidden rounded-lg">
              <img
                src={anime.coverImage.large}
                alt={anime.title.romaji}
                className="w-full aspect-[3/4] object-cover group-hover:scale-110 transition-transform duration-300"
              />
            </div>
            <p className="text-xs text-white font-medium line-clamp-2">
              {anime.title.english || anime.title.romaji}
            </p>
            <p className="text-xs text-gray-400">
              {anime.seasonYear || anime.startDate?.year}年
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
