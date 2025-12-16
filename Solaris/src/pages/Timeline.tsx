import React, { useState, useRef } from "react";
import {
  Clock,
  Calendar,
  Loader2,
  Film,
  Star,
  ExternalLink,
} from "lucide-react";
import { QuickIDSelector } from "../components/QuickIDSelector";
import { StatsPanel, BirthYearCard } from "../components/timeline";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface Anime {
  id: number;
  title: {
    romaji: string;
    english: string | null;
  };
  coverImage: {
    large: string;
  };
  averageScore: number;
  popularity: number;
  genres: string[];
  is_watched?: boolean;
  user_score?: number;
  selection_reason?: string;
}

interface TimelineMilestone {
  age: number;
  year: number;
  label: string;
  anime: Anime[];
}

interface ChronologicalEntry {
  year: number;
  age: number;
  anime: Anime;
}

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

interface TimelineResponse {
  username: string;
  birth_year: number;
  timeline_data: TimelineMilestone[];
  chronological_data: ChronologicalEntry[];
  stats?: TimelineStats;
  birth_year_anime?: BirthYearAnime[];
}

export const Timeline = () => {
  const [birthYear, setBirthYear] = useState<string>("");
  const [birthMonth, setBirthMonth] = useState<string>("");
  const [birthDay, setBirthDay] = useState<string>("");
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [checkingUser, setCheckingUser] = useState(false);
  const [isDateLocked, setIsDateLocked] = useState(false);
  const [result, setResult] = useState<TimelineResponse | null>(null);
  const [error, setError] = useState("");
  const [yearError, setYearError] = useState("");
  const [monthError, setMonthError] = useState("");
  const [dayError, setDayError] = useState("");
  const dateInputRef = useRef<HTMLInputElement>(null);

  // 即時驗證：年份
  const handleYearChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBirthYear(e.target.value);
    const now = new Date();
    const year = parseInt(e.target.value);
    if (
      !e.target.value ||
      isNaN(year) ||
      year < 1900 ||
      year > now.getFullYear()
    ) {
      setYearError("年份格式錯誤");
    } else {
      setYearError("");
    }
  };
  // 即時驗證：月份（允許前導0，但不能超過兩位數）
  const handleMonthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setBirthMonth(val);
    const month = parseInt(val, 10);
    if (!val || isNaN(month) || month < 1 || month > 12 || val.length > 2) {
      setMonthError("月份格式錯誤");
    } else {
      setMonthError("");
    }
  };
  // 即時驗證：日期（允許前導0，但不能超過兩位數）
  const handleDayChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setBirthDay(val);
    const day = parseInt(val, 10);
    if (!val || isNaN(day) || day < 1 || day > 31 || val.length > 2) {
      setDayError("日期格式錯誤");
    } else {
      setDayError("");
    }
  };

  const checkUser = async () => {
    if (!username.trim()) {
      setIsDateLocked(false);
      return;
    }

    setCheckingUser(true);
    try {
      const response = await fetch(`${BACKEND_URL}/user_info`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: username.trim() }),
      });

      if (response.ok) {
        const data = await response.json();
        const dob = data.dateOfBirth;
        if (dob && dob.year) {
          // pad month/day to 2 digits
          setBirthYear(dob.year.toString());
          setBirthMonth(dob.month ? dob.month.toString().padStart(2, "0") : "");
          setBirthDay(dob.day ? dob.day.toString().padStart(2, "0") : "");
          setIsDateLocked(true);
          setError("");
        } else {
          setIsDateLocked(false);
        }
      } else {
        setIsDateLocked(false);
      }
    } catch (err) {
      console.error(err);
      setIsDateLocked(false);
    } finally {
      setCheckingUser(false);
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();

    setError("");
    setYearError("");
    setMonthError("");
    setDayError("");

    let valid = true;
    const now = new Date();
    const year = parseInt(birthYear);
    const month = parseInt(birthMonth);
    const day = parseInt(birthDay);

    if (!birthYear || isNaN(year) || year < 1900 || year > now.getFullYear()) {
      setYearError("年份格式錯誤");
      valid = false;
    }
    if (!birthMonth || isNaN(month) || month < 1 || month > 12) {
      setMonthError("月份格式錯誤");
      valid = false;
    }
    if (!birthDay || isNaN(day) || day < 1 || day > 31) {
      setDayError("日期格式錯誤");
      valid = false;
    }
    const testDate = new Date(year, month - 1, day);
    if (
      !isNaN(year) &&
      !isNaN(month) &&
      !isNaN(day) &&
      (testDate.getFullYear() !== year ||
        testDate.getMonth() + 1 !== month ||
        testDate.getDate() !== day)
    ) {
      setDayError("日期不存在");
      valid = false;
    }
    if (!valid) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${BACKEND_URL}/timeline`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: username.trim() || null,
          birth_year: year,
          birth_month: month,
          birth_day: day,
        }),
      });

      if (!response.ok) {
        throw new Error("無法生成時間軸");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("發生錯誤，請稍後再試");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="max-w-5xl mx-auto px-4"
      style={{
        transform: "none",
        overflow: "visible",
        position: "static",
        filter: "none",
        zIndex: "auto",
      }}
    >
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-600 text-transparent bg-clip-text flex items-center justify-center gap-3">
          <Clock className="w-10 h-10 text-amber-500" />
          動畫大世紀
        </h1>
        <p className="text-gray-400">
          輸入你的出生年份，回顧你成長過程中的霸權動畫
        </p>
      </div>

      <form
        onSubmit={handleGenerate}
        className="mb-16 bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-xl max-w-2xl mx-auto"
      >
        <div className="space-y-6">
          {/* Row 1: ID Input */}
          <div className="relative">
            <QuickIDSelector
              value={username}
              onChange={(value) => {
                setUsername(value);
                if (!value) setIsDateLocked(false);
              }}
              label="AniList 使用者名稱（選填）"
              placeholder="輸入使用者名稱客製化內容（例如: senba1000m3）"
              required={false}
            />
            {checkingUser && (
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <Loader2 className="w-5 h-5 animate-spin text-amber-500" />
              </div>
            )}
            <p className="text-xs text-gray-500 mt-1">
              若找不到 ID 或無生日資料，請手動輸入下方日期
            </p>
          </div>

          <div className="flex items-center justify-between mb-1 relative">
            <span className="block text-sm font-medium text-gray-300">
              出生日期
              {isDateLocked && (
                <span className="text-amber-500 text-xs ml-2">
                  (已從 ID 鎖定)
                </span>
              )}
            </span>
            <div className="relative inline-block">
              <button
                type="button"
                className="text-gray-400 hover:text-amber-500 relative z-10 pointer-events-none"
                tabIndex={-1}
                disabled={isDateLocked}
                aria-label="用日曆選擇"
              >
                <Calendar className="w-5 h-5" />
              </button>
              <input
                type="date"
                ref={dateInputRef}
                className="date-picker-dark"
                style={{
                  position: "absolute",
                  right: "-8px",
                  top: "-8px",
                  opacity: 0,
                  width: "40px",
                  height: "40px",
                  cursor: isDateLocked ? "not-allowed" : "pointer",
                  colorScheme: "dark",
                }}
                max={new Date().toISOString().split("T")[0]}
                disabled={isDateLocked}
                onChange={(e) => {
                  const [y, m, d] = e.target.value.split("-");
                  setBirthYear(y);
                  setBirthMonth(m);
                  setBirthDay(d);
                  setYearError("");
                  setMonthError("");
                  setDayError("");
                }}
              />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4 relative">
            <div className="relative">
              <input
                type="number"
                value={birthYear}
                onChange={handleYearChange}
                disabled={isDateLocked}
                placeholder="YYYY"
                className={`appearance-none w-full px-4 py-3 bg-gray-700 rounded-lg border ${
                  yearError ? "border-red-500" : "border-gray-600"
                } focus:border-amber-500 focus:ring-2 focus:ring-amber-500 outline-none transition-all text-lg ${
                  isDateLocked ? "opacity-50 cursor-not-allowed" : ""
                }`}
              />
              <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 text-sm">
                年
              </span>
              {yearError && (
                <div className="text-red-500 text-xs mt-1">{yearError}</div>
              )}
            </div>
            <div className="relative">
              <input
                type="number"
                value={birthMonth}
                onChange={handleMonthChange}
                disabled={isDateLocked}
                placeholder="MM"
                className={`appearance-none w-full px-4 py-3 bg-gray-700 rounded-lg border ${
                  monthError ? "border-red-500" : "border-gray-600"
                } focus:border-amber-500 focus:ring-2 focus:ring-amber-500 outline-none transition-all text-lg ${
                  isDateLocked ? "opacity-50 cursor-not-allowed" : ""
                }`}
              />
              <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 text-sm">
                月
              </span>
              {monthError && (
                <div className="text-red-500 text-xs mt-1">{monthError}</div>
              )}
            </div>
            <div className="relative">
              <input
                type="number"
                value={birthDay}
                onChange={handleDayChange}
                disabled={isDateLocked}
                placeholder="DD"
                className={`appearance-none w-full px-4 py-3 bg-gray-700 rounded-lg border ${
                  dayError ? "border-red-500" : "border-gray-600"
                } focus:border-amber-500 focus:ring-2 focus:ring-amber-500 outline-none transition-all text-lg ${
                  isDateLocked ? "opacity-50 cursor-not-allowed" : ""
                }`}
              />
              <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 text-sm">
                日
              </span>
              {dayError && (
                <div className="text-red-500 text-xs mt-1">{dayError}</div>
              )}
            </div>
          </div>
        </div>
        <button
          type="submit"
          disabled={
            loading ||
            !!yearError ||
            !!monthError ||
            !!dayError ||
            !birthYear ||
            !birthMonth ||
            !birthDay
          }
          className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 rounded-lg font-bold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
        >
          {loading ? (
            <>
              <Loader2 className="w-6 h-6 animate-spin" />
              正在穿越時空...
            </>
          ) : (
            <>
              <Film className="w-6 h-6" />
              生成回憶錄
            </>
          )}
        </button>
      </form>

      {error && (
        <div className="text-center text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800 max-w-2xl mx-auto">
          {error}
        </div>
      )}

      {result &&
        result.chronological_data &&
        result.chronological_data.length > 0 && (
          <div className="space-y-16 pb-20">
            {/* Stats & Birth Year Anime Section */}
            {(result.stats ||
              (result.birth_year_anime &&
                result.birth_year_anime.length > 0)) && (
              <div className="grid grid-cols-[1fr_2fr] md:grid-cols-[1fr_2fr] gap-8">
                {/* Stats */}
                {result.stats && Object.keys(result.stats).length > 0 && (
                  <StatsPanel stats={result.stats} />
                )}

                {/* Birth Year Anime */}
                {result.birth_year_anime &&
                  result.birth_year_anime.length > 0 && (
                    <BirthYearCard
                      birthYear={result.birth_year}
                      animeList={result.birth_year_anime}
                    />
                  )}
              </div>
            )}

            {/* Chronological Flow Section */}
            <div>
              <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-2">
                <Calendar className="w-6 h-6 text-amber-500" />
                時光迴廊 ({result.birth_year} - {new Date().getFullYear()})
              </h2>

              {/* Birth Year - Centered */}
              <div className="flex justify-center mb-8">
                <div
                  className={`w-full max-w-md bg-gray-800 rounded-xl overflow-hidden border-4 shadow-2xl group cursor-pointer transform hover:scale-105 transition-all duration-300 ${
                    result.chronological_data[0].anime.is_watched
                      ? "border-amber-500 border-[6px] shadow-[0_0_20px_rgba(245,158,11,0.4)]"
                      : "border-gray-700"
                  }`}
                  onClick={() =>
                    window.open(
                      `https://anilist.co/anime/${result.chronological_data[0].anime.id}`,
                      "_blank",
                    )
                  }
                >
                  <div className="relative h-64">
                    <img
                      src={result.chronological_data[0].anime.coverImage?.large}
                      alt={result.chronological_data[0].anime.title.romaji}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent" />
                    <div className="absolute bottom-4 left-4 right-4">
                      <div className="flex items-center justify-between mb-2">
                        <a
                          href={`https://anilist.co/search/anime?year=${result.chronological_data[0].year}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="bg-amber-500 text-black font-bold px-3 py-1 rounded-full text-sm hover:bg-amber-400 transition-colors flex items-center gap-1 z-20"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {result.chronological_data[0].year} 誕生{" "}
                          <ExternalLink className="w-3 h-3" />
                        </a>
                        {result.chronological_data[0].anime.is_watched && (
                          <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                            <Star className="w-3 h-3 fill-current" />
                            {result.chronological_data[0].anime.user_score}
                          </span>
                        )}
                      </div>
                      <h3 className="text-xl font-bold text-white line-clamp-1">
                        {result.chronological_data[0].anime.title.english ||
                          result.chronological_data[0].anime.title.romaji}
                      </h3>
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-gray-400 text-sm mb-2">
                      {result.chronological_data[0].anime.selection_reason}
                    </p>
                    <div className="flex gap-2">
                      {result.chronological_data[0].anime.genres
                        .slice(0, 3)
                        .map((g) => (
                          <span
                            key={g}
                            className="text-xs bg-gray-700 px-2 py-1 rounded text-gray-300"
                          >
                            {g}
                          </span>
                        ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Subsequent Years Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {result.chronological_data.slice(1).map((entry) => (
                  <div
                    key={entry.year}
                    className={`bg-gray-800 rounded-lg overflow-hidden border-2 shadow-lg group cursor-pointer hover:transform hover:-translate-y-1 transition-all duration-300 ${
                      entry.anime.is_watched
                        ? "border-amber-500 border-[6px] shadow-[0_0_15px_rgba(245,158,11,0.3)]"
                        : "border-gray-700 hover:border-gray-500"
                    }`}
                    onClick={() =>
                      window.open(
                        `https://anilist.co/anime/${entry.anime.id}`,
                        "_blank",
                      )
                    }
                  >
                    <div className="relative h-40">
                      <img
                        src={entry.anime.coverImage?.large}
                        alt={entry.anime.title.romaji}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute top-2 left-2 z-10">
                        <a
                          href={`https://anilist.co/search/anime?year=${entry.year}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="bg-gray-900/80 text-white text-xs font-bold px-2 py-1 rounded backdrop-blur-sm hover:bg-amber-500 hover:text-black transition-colors flex items-center gap-1"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {entry.year} <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                      {entry.anime.is_watched && (
                        <div className="absolute top-2 right-2">
                          <span className="bg-green-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded flex items-center gap-1 shadow-sm">
                            <Star className="w-2.5 h-2.5 fill-current" />
                            {entry.anime.user_score}
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="p-3">
                      <h4 className="font-bold text-sm text-white line-clamp-1 mb-1 group-hover:text-amber-400 transition-colors">
                        {entry.anime.title.english || entry.anime.title.romaji}
                      </h4>
                      <p className="text-xs text-gray-500 line-clamp-1 mb-2">
                        {entry.anime.selection_reason}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Milestones Section (Original Timeline) */}
            <div className="relative pt-8 border-t border-gray-700">
              <h2 className="text-2xl font-bold text-white mb-12 flex items-center gap-2 justify-center">
                <Clock className="w-6 h-6 text-amber-500" />
                成長里程碑
              </h2>

              {/* Vertical Line */}
              <div className="absolute left-4 md:left-1/2 top-24 bottom-0 w-1 bg-gray-700 transform -translate-x-1/2 hidden md:block" />

              {result.timeline_data.map((milestone, index) => (
                <div
                  key={milestone.age}
                  className={`relative flex flex-col md:flex-row gap-8 items-center mb-12 ${
                    index % 2 === 0 ? "md:flex-row-reverse" : ""
                  }`}
                >
                  {/* Timeline Dot */}
                  <div className="absolute left-4 md:left-1/2 w-8 h-8 bg-gray-900 border-4 border-amber-500 rounded-full transform -translate-x-1/2 z-10 hidden md:flex items-center justify-center">
                    <div className="w-2 h-2 bg-amber-500 rounded-full" />
                  </div>

                  {/* Content Card */}
                  <div className="w-full md:w-1/2 pl-12 md:pl-0">
                    <div
                      className={`bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg hover:border-amber-500/50 transition-colors ${
                        index % 2 === 0 ? "md:mr-12" : "md:ml-12"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-4 border-b border-gray-700 pb-2">
                        <div>
                          <h3 className="text-2xl font-bold text-amber-400">
                            {milestone.year}
                          </h3>
                          <p className="text-gray-400 text-sm font-mono">
                            {milestone.label}
                          </p>
                        </div>
                        <Calendar className="w-6 h-6 text-gray-500" />
                      </div>

                      <div className="space-y-4">
                        {milestone.anime.map((anime) => (
                          <div
                            key={anime.id}
                            className={`flex gap-4 group cursor-pointer p-2 rounded-lg transition-colors ${
                              anime.is_watched
                                ? "bg-gray-700/50 border border-amber-500/30"
                                : "hover:bg-gray-700/30"
                            }`}
                            onClick={() =>
                              window.open(
                                `https://anilist.co/anime/${anime.id}`,
                                "_blank",
                              )
                            }
                          >
                            <div className="relative flex-shrink-0 w-16 h-24 overflow-hidden rounded-md">
                              <img
                                src={anime.coverImage?.large}
                                alt={anime.title.romaji}
                                className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                              />
                              {anime.is_watched && (
                                <div className="absolute top-1 right-1">
                                  <span className="bg-green-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded flex items-center gap-1 shadow-sm">
                                    <Star className="w-2.5 h-2.5 fill-current" />
                                    {anime.user_score}
                                  </span>
                                </div>
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="font-semibold text-white truncate group-hover:text-amber-400 transition-colors">
                                {anime.title.english || anime.title.romaji}
                              </h4>
                              <p className="text-xs text-gray-400 mb-1">
                                {anime.selection_reason}
                              </p>
                              <div className="flex flex-wrap gap-1 mt-1 mb-2">
                                {anime.genres.slice(0, 2).map((g) => (
                                  <span
                                    key={g}
                                    className="text-[10px] px-1.5 py-0.5 bg-gray-700 rounded text-gray-300"
                                  >
                                    {g}
                                  </span>
                                ))}
                              </div>
                              <div className="flex items-center gap-3 text-xs text-gray-400">
                                <div className="flex items-center gap-1">
                                  <Star className="w-3 h-3 text-yellow-500" />
                                  {anime.averageScore}%
                                </div>
                                <div>❤ {anime.popularity.toLocaleString()}</div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
    </div>
  );
};
