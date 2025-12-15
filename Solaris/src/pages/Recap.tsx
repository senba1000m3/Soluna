import React, { useState, useEffect } from "react";
import {
  Trophy,
  Calendar,
  Clock,
  TrendingUp,
  Play,
  Loader2,
  Star,
  Eye,
  BarChart3,
} from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  tier: "bronze" | "silver" | "gold" | "diamond";
}

interface RepeatAnime {
  id: number;
  title: string;
  title_english: string | null;
  coverImage: string;
  repeat_count: number;
  score: number;
}

interface MonthlyRepresentative {
  month: number;
  id: number;
  title: string;
  title_english: string | null;
  coverImage: string;
  score: number;
}

interface StudioInfo {
  id: number;
  name: string;
  siteUrl: string;
  count: number;
}

interface VoiceActorInfo {
  id: number;
  name: string;
  native: string;
  image: string;
  siteUrl: string;
  count: number;
}

interface TopAnime {
  id: number;
  title: string;
  title_english: string | null;
  coverImage: string;
  score: number;
  episodes: number;
  progress: number;
  status: string;
  genres: string[];
  format: string;
  averageScore: number;
  year: number | null;
}

interface RecapData {
  username: string;
  year: number | null;
  is_all_time: boolean;
  total_anime: number;
  total_episodes: number;
  total_minutes: number;
  total_hours: number;
  completed_count: number;
  watching_count: number;
  dropped_count: number;
  planned_count: number;
  paused_count: number;
  repeating_count: number;
  top_anime: TopAnime[];
  genre_distribution: Record<string, number>;
  format_distribution: Record<string, number>;
  tag_distribution: Record<string, number>;
  studio_distribution: Record<string, StudioInfo>;
  voice_actor_distribution: Record<string, VoiceActorInfo>;
  season_distribution: Record<string, number>;
  month_added_distribution: Record<string, number>;
  month_completed_distribution: Record<string, number>;
  monthly_representative: Record<string, MonthlyRepresentative>;
  most_rewatched: RepeatAnime[];
  average_score: number;
  total_scored: number;
  achievements: Achievement[];
}

interface AnimationSlide {
  type: "stats" | "top_anime" | "genres" | "achievements" | "final";
  content: React.ReactNode;
}

export const Recap = () => {
  const [username, setUsername] = useState("");
  const [selectedYear, setSelectedYear] = useState<string>("all");
  const [recapData, setRecapData] = useState<RecapData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAnimation, setShowAnimation] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [animationComplete, setAnimationComplete] = useState(false);

  // Generate year options (current year back to 2010)
  const currentYear = new Date().getFullYear();
  const yearOptions = ["all"];
  for (let year = currentYear; year >= 2010; year--) {
    yearOptions.push(year.toString());
  }

  const fetchRecap = async () => {
    console.log("=== Recap 請求開始 ===");
    console.log("Username:", username);
    console.log("Selected Year:", selectedYear);

    if (!username.trim()) {
      console.error("錯誤: 使用者名稱為空");
      setError("請輸入 AniList 使用者名稱");
      return;
    }

    setLoading(true);
    setError("");
    setRecapData(null);
    setAnimationComplete(false);

    try {
      const payload: any = {
        username: username.trim(),
      };

      if (selectedYear !== "all") {
        payload.year = parseInt(selectedYear);
      }

      console.log("發送 Recap 請求...");
      console.log("Payload:", payload);
      console.log("Backend URL:", BACKEND_URL);

      const response = await fetch(`${BACKEND_URL}/recap`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      console.log("收到回應，狀態碼:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API 錯誤回應:", errorText);
        throw new Error(`無法取得 Recap 數據 (${response.status})`);
      }

      const data: RecapData = await response.json();
      console.log("成功取得 Recap 數據:");
      console.log("- Total Anime:", data.total_anime);
      console.log("- Total Episodes:", data.total_episodes);
      console.log("- Total Hours:", data.total_hours);
      console.log("- Achievements:", data.achievements.length);
      console.log("完整數據:", data);

      setRecapData(data);
      setShowAnimation(true);
      setCurrentSlide(0);
      console.log("開始播放動畫");
    } catch (err: any) {
      console.error("=== Recap 請求失敗 ===");
      console.error("錯誤訊息:", err.message);
      console.error("錯誤詳情:", err);
      setError(err.message || "發生錯誤，請檢查使用者名稱或稍後再試");
    } finally {
      setLoading(false);
      console.log("=== Recap 請求結束 ===");
    }
  };

  const replayAnimation = () => {
    setShowAnimation(true);
    setCurrentSlide(0);
    setAnimationComplete(false);
  };

  const skipAnimation = () => {
    setShowAnimation(false);
    setAnimationComplete(true);
  };

  // Auto-advance animation slides
  useEffect(() => {
    if (showAnimation && recapData) {
      const slides = generateAnimationSlides(recapData);
      if (currentSlide < slides.length - 1) {
        const timer = setTimeout(() => {
          setCurrentSlide((prev) => prev + 1);
        }, 3000); // 3 seconds per slide
        return () => clearTimeout(timer);
      } else {
        // Animation complete
        const timer = setTimeout(() => {
          setShowAnimation(false);
          setAnimationComplete(true);
        }, 3000);
        return () => clearTimeout(timer);
      }
    }
  }, [showAnimation, currentSlide, recapData]);

  // Keyboard navigation for slides
  useEffect(() => {
    if (!showAnimation || !recapData) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const slides = generateAnimationSlides(recapData);

      if (e.key === "ArrowLeft" && currentSlide > 0) {
        setCurrentSlide((prev) => prev - 1);
      } else if (e.key === "ArrowRight" && currentSlide < slides.length - 1) {
        setCurrentSlide((prev) => prev + 1);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [showAnimation, currentSlide, recapData]);

  const generateAnimationSlides = (data: RecapData): AnimationSlide[] => {
    const slides: AnimationSlide[] = [];

    // Slide 1: Welcome
    slides.push({
      type: "stats",
      content: (
        <div className="text-center animate-fade-in">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            {data.is_all_time ? "總覽 Recap" : `${data.year} Recap`}
          </h1>
          <p className="text-2xl text-gray-300">{data.username} 的動漫回顧</p>
        </div>
      ),
    });

    // Slide 2: Total Stats
    slides.push({
      type: "stats",
      content: (
        <div className="text-center animate-fade-in">
          <h2 className="text-4xl font-bold mb-8 text-purple-300">
            你的觀看紀錄
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-gradient-to-br from-purple-600 to-purple-800 p-8 rounded-2xl">
              <div className="text-6xl font-bold mb-2">{data.total_anime}</div>
              <div className="text-xl text-purple-200">部動漫</div>
            </div>
            <div className="bg-gradient-to-br from-pink-600 to-pink-800 p-8 rounded-2xl">
              <div className="text-6xl font-bold mb-2">
                {data.total_episodes}
              </div>
              <div className="text-xl text-pink-200">集數</div>
            </div>
            <div className="bg-gradient-to-br from-blue-600 to-blue-800 p-8 rounded-2xl">
              <div className="text-6xl font-bold mb-2">{data.total_hours}</div>
              <div className="text-xl text-blue-200">小時</div>
            </div>
          </div>
        </div>
      ),
    });

    // Slide 3: Completion Stats
    if (data.completed_count > 0) {
      slides.push({
        type: "stats",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-green-300">完成紀錄</h2>
            <div className="text-7xl font-bold mb-4 text-green-400">
              {data.completed_count}
            </div>
            <p className="text-2xl text-gray-300">部動漫完整看完！</p>
            {data.average_score > 0 && (
              <div className="mt-8">
                <p className="text-xl text-gray-400">平均評分</p>
                <p className="text-5xl font-bold text-yellow-400 mt-2">
                  {data.average_score}
                </p>
              </div>
            )}
          </div>
        ),
      });
    }

    // Slide 4: Top Anime
    if (data.top_anime.length > 0) {
      slides.push({
        type: "top_anime",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-purple-300">
              你的最愛 Top 5
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {data.top_anime.slice(0, 5).map((anime, index) => (
                <div
                  key={anime.id}
                  className="relative group animate-scale-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="absolute -top-3 -left-3 bg-gradient-to-br from-yellow-400 to-yellow-600 text-black font-bold text-xl w-10 h-10 rounded-full flex items-center justify-center z-10 shadow-lg">
                    {index + 1}
                  </div>
                  <img
                    src={anime.coverImage}
                    alt={anime.title}
                    className="w-full h-64 object-cover rounded-xl shadow-2xl"
                  />
                  <div className="mt-2 text-sm font-semibold line-clamp-2">
                    {anime.title_english || anime.title}
                  </div>
                  <div className="flex items-center justify-center gap-1 mt-1">
                    <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                    <span className="text-yellow-400 font-bold">
                      {anime.score}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ),
      });
    }

    // Slide 5: Top Genre
    if (Object.keys(data.genre_distribution).length > 0) {
      const topGenre = Object.entries(data.genre_distribution)[0];
      slides.push({
        type: "genres",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-purple-300">
              你最愛的類型
            </h2>
            <div className="text-8xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text py-3">
              {topGenre[0]}
            </div>
            <p className="text-3xl text-gray-300 mt-16">
              觀看了 {topGenre[1]} 部
            </p>
          </div>
        ),
      });
    }

    // Slide 5.5: Top Studio
    if (Object.keys(data.studio_distribution).length > 0) {
      const topStudio = Object.values(data.studio_distribution)[0];
      slides.push({
        type: "genres",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-pink-300">
              最愛的製作公司
            </h2>
            <div className="text-7xl font-bold mb-4 bg-gradient-to-r from-pink-400 to-rose-600 text-transparent bg-clip-text py-3">
              {topStudio.name}
            </div>
            <p className="text-3xl text-gray-300 mt-16">
              製作了 {topStudio.count} 部你看過的動漫
            </p>
          </div>
        ),
      });
    }

    // Slide 5.6: Top Voice Actor
    if (Object.keys(data.voice_actor_distribution).length > 0) {
      const topVA = Object.values(data.voice_actor_distribution)[0];
      slides.push({
        type: "genres",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-blue-300">
              最常聽到的聲優
            </h2>
            {topVA.image && (
              <img
                src={topVA.image}
                alt={topVA.name}
                className="w-48 h-48 rounded-full mx-auto mb-6 border-4 border-blue-400 shadow-2xl object-cover"
              />
            )}
            <div className="text-6xl font-bold mb-2 text-white">
              {topVA.name}
            </div>
            {topVA.native && (
              <div className="text-3xl mb-4 text-blue-200">{topVA.native}</div>
            )}
            <p className="text-3xl text-gray-300">
              在 {topVA.count} 部作品中獻聲
            </p>
          </div>
        ),
      });
    }

    // Slide 5.7: Most Rewatched
    if (data.most_rewatched.length > 0) {
      const topRewatch = data.most_rewatched[0];
      slides.push({
        type: "genres",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-purple-300">
              你的最愛作品
            </h2>
            <img
              src={topRewatch.coverImage}
              alt={topRewatch.title}
              className="w-64 h-96 mx-auto rounded-2xl shadow-2xl mb-6 object-cover"
            />
            <div className="text-3xl font-bold mb-2 text-white">
              {topRewatch.title_english || topRewatch.title}
            </div>
            <p className="text-5xl font-bold text-purple-400 mb-2">
              重看了 {topRewatch.repeat_count} 次！
            </p>
            <p className="text-xl text-gray-300">真的超愛這部呢</p>
          </div>
        ),
      });
    }

    // Slide 6: Season Statistics
    if (Object.keys(data.season_distribution).length > 0) {
      const topSeason = Object.entries(data.season_distribution)[0];
      slides.push({
        type: "stats",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-green-300">
              最熱血的季節
            </h2>
            <div className="text-7xl font-bold mb-4 bg-gradient-to-r from-green-400 to-emerald-600 text-transparent bg-clip-text py-3">
              {topSeason[0]}
            </div>
            <p className="text-3xl text-gray-300 mt-16">
              這一季追了 {topSeason[1]} 部動漫！
            </p>
          </div>
        ),
      });
    }

    // Slide 7: Tag Statistics
    if (Object.keys(data.tag_distribution).length > 0) {
      const topTag = Object.entries(data.tag_distribution)[0];
      slides.push({
        type: "genres",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-cyan-300">
              最感興趣的主題
            </h2>
            <div className="text-8xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-blue-600 text-transparent bg-clip-text px-4 py-3">
              {topTag[0]}
            </div>
            <p className="text-3xl text-gray-300 mt-16">
              出現在 {topTag[1]} 部作品中
            </p>
          </div>
        ),
      });
    }

    // Slide 8: Overview - Top 3 in each category (2 columns with scrolling)
    slides.push({
      type: "final",
      content: (
        <div className="animate-fade-in px-4 h-full flex flex-col">
          <h2 className="text-5xl font-bold mb-16 text-center bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            你的年度回顧總覽
          </h2>
          <div className="flex-1 overflow-y-auto pr-2">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-7xl mx-auto">
              {/* Top 3 Anime */}
              {data.top_anime.length > 0 && (
                <div className="bg-gray-800/50 p-6 rounded-2xl border border-purple-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <span className="text-2xl">👑</span>
                    <span>最愛作品 Top 3</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-3">
                    {data.top_anime.slice(0, 3).map((anime, index) => (
                      <div key={anime.id} className="text-center">
                        <div className="relative inline-block mb-2">
                          <img
                            src={anime.coverImage}
                            alt={anime.title}
                            className="w-24 h-32 object-cover rounded-lg shadow-lg"
                          />
                          <div className="absolute -top-2 -left-2 bg-gradient-to-br from-yellow-400 to-yellow-600 text-black font-bold text-base w-6 h-6 rounded-full flex items-center justify-center shadow-lg">
                            {index + 1}
                          </div>
                        </div>
                        <p className="text-xs font-semibold line-clamp-2 px-1">
                          {anime.title_english || anime.title}
                        </p>
                        <p className="text-yellow-400 font-bold text-xs mt-1">
                          ⭐ {anime.score}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Top 3 Genres */}
              {Object.keys(data.genre_distribution).length > 0 && (
                <div className="bg-gray-800/50 p-6 rounded-2xl border border-pink-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <span className="text-2xl">🎭</span>
                    <span>最愛類型 Top 3</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-3">
                    {Object.entries(data.genre_distribution)
                      .slice(0, 3)
                      .map(([genre, count], index) => (
                        <div
                          key={genre}
                          className="bg-gradient-to-br from-pink-600/30 to-rose-800/30 p-3 rounded-xl text-center border border-pink-500/50"
                        >
                          <div className="text-2xl font-bold text-pink-300 mb-1">
                            #{index + 1}
                          </div>
                          <div className="text-sm font-bold text-white mb-1 line-clamp-1">
                            {genre}
                          </div>
                          <div className="text-xs text-pink-200">
                            {count} 部
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Top 3 Studios */}
              {Object.keys(data.studio_distribution).length > 0 && (
                <div className="bg-gray-800/50 p-6 rounded-2xl border border-blue-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <span className="text-2xl">🏢</span>
                    <span>最愛製作公司 Top 3</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-3">
                    {Object.values(data.studio_distribution)
                      .slice(0, 3)
                      .map((studio, index) => (
                        <div
                          key={studio.id}
                          className="bg-gradient-to-br from-blue-600/30 to-blue-800/30 p-3 rounded-xl text-center border border-blue-500/50"
                        >
                          <div className="text-2xl font-bold text-blue-300 mb-1">
                            #{index + 1}
                          </div>
                          <div className="text-sm font-bold text-white mb-1 line-clamp-2 h-10">
                            {studio.name}
                          </div>
                          <div className="text-xs text-blue-200">
                            {studio.count} 部
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Top 3 Voice Actors */}
              {Object.keys(data.voice_actor_distribution).length > 0 && (
                <div className="bg-gray-800/50 p-6 rounded-2xl border border-cyan-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <span className="text-2xl">🎤</span>
                    <span>最愛聲優 Top 3</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.values(data.voice_actor_distribution)
                      .slice(0, 3)
                      .map((va, index) => (
                        <div key={va.id} className="text-center">
                          <div className="relative inline-block mb-2">
                            {va.image ? (
                              <img
                                src={va.image}
                                alt={va.name}
                                className="w-20 h-20 object-cover rounded-full shadow-lg border-4 border-cyan-400"
                              />
                            ) : (
                              <div className="w-20 h-20 bg-cyan-600/30 rounded-full flex items-center justify-center border-4 border-cyan-400">
                                <span className="text-3xl">👤</span>
                              </div>
                            )}
                            <div className="absolute -top-1 -right-1 bg-cyan-500 text-white font-bold text-sm w-6 h-6 rounded-full flex items-center justify-center shadow-lg">
                              {index + 1}
                            </div>
                          </div>
                          <p className="text-xs font-bold text-white line-clamp-2 px-1 h-8">
                            {va.name}
                          </p>
                          <p className="text-xs text-cyan-200 line-clamp-1 px-1">
                            {va.native}
                          </p>
                          <p className="text-cyan-400 font-bold text-xs mt-1">
                            {va.count} 部
                          </p>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Top 3 Tags */}
              {Object.keys(data.tag_distribution).length > 0 && (
                <div className="bg-gray-800/50 p-6 rounded-2xl border border-green-500/30">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <span className="text-2xl">🏷️</span>
                    <span>最愛主題 Top 3</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-3">
                    {Object.entries(data.tag_distribution)
                      .slice(0, 3)
                      .map(([tag, count], index) => (
                        <div
                          key={tag}
                          className="bg-gradient-to-br from-green-600/30 to-green-800/30 p-3 rounded-xl text-center border border-green-500/50"
                        >
                          <div className="text-2xl font-bold text-green-300 mb-1">
                            #{index + 1}
                          </div>
                          <div className="text-sm font-bold text-white mb-1 line-clamp-2 h-10">
                            {tag}
                          </div>
                          <div className="text-xs text-green-200">
                            {count} 部
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ),
    });

    // Slide 9: Achievements
    {
      /* Achievements */
    }
    if (data.achievements.length > 0) {
      slides.push({
        type: "achievements",
        content: (
          <div className="text-center animate-fade-in">
            <h2 className="text-4xl font-bold mb-8 text-yellow-300">
              成就解鎖！
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
              {data.achievements.map((achievement, index) => {
                const tierColors = {
                  bronze: "from-orange-700 to-orange-900",
                  silver: "from-gray-400 to-gray-600",
                  gold: "from-yellow-500 to-yellow-700",
                  diamond: "from-cyan-400 to-blue-600",
                };
                const tierColor =
                  tierColors[achievement.tier] || tierColors.bronze;

                return (
                  <div
                    key={achievement.id}
                    className={`bg-gradient-to-br ${tierColor} p-6 rounded-2xl shadow-xl animate-scale-in`}
                    style={{ animationDelay: `${index * 0.15}s` }}
                  >
                    <div className="text-5xl mb-3">{achievement.icon}</div>
                    <h3 className="text-xl font-bold mb-2">
                      {achievement.title}
                    </h3>
                    <p className="text-white opacity-90">
                      {achievement.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        ),
      });
    }

    // Final Slide
    slides.push({
      type: "final",
      content: (
        <div className="text-center animate-fade-in">
          <h2 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            感謝你的陪伴！
          </h2>
          <p className="text-2xl text-gray-300 mb-8">
            {data.is_all_time
              ? "這是你的動漫旅程總覽"
              : `這是你在 ${data.year} 年的動漫旅程`}
          </p>
          <div className="text-6xl">🎉</div>
        </div>
      ),
    });

    return slides;
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Animation View */}
      {showAnimation && recapData && (
        <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 z-50 flex items-center justify-center p-8">
          <div className="max-w-6xl w-full">
            {generateAnimationSlides(recapData)[currentSlide]?.content}
          </div>
          <div className="absolute top-4 right-4 flex gap-2">
            <button
              onClick={skipAnimation}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-sm"
            >
              跳過動畫
            </button>
          </div>
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex items-center gap-4">
            <button
              onClick={() =>
                currentSlide > 0 && setCurrentSlide((prev) => prev - 1)
              }
              disabled={currentSlide === 0}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ← 上一張
            </button>
            <div className="flex gap-2">
              {generateAnimationSlides(recapData).map((_, index) => (
                <div
                  key={index}
                  className={`w-3 h-3 rounded-full transition-all ${
                    index === currentSlide ? "bg-purple-500 w-8" : "bg-gray-600"
                  }`}
                />
              ))}
            </div>
            <button
              onClick={() => {
                const slides = generateAnimationSlides(recapData);
                if (currentSlide < slides.length - 1) {
                  setCurrentSlide((prev) => prev + 1);
                }
              }}
              disabled={
                currentSlide === generateAnimationSlides(recapData).length - 1
              }
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              下一張 →
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
          動漫動態回顧
        </h1>
        <p className="text-gray-400 text-lg">
          回顧你的動漫觀看旅程，發現你的觀影習慣
        </p>
      </div>

      {/* Input Form */}
      {!animationComplete && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            fetchRecap();
          }}
          className="mb-12 bg-gray-800 p-8 rounded-xl border border-gray-700 max-w-3xl mx-auto"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                AniList 使用者名稱
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="例如: senba1000m3"
                className="w-full px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all text-lg"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                選擇模式
              </label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="w-full px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all text-lg"
              >
                <option value="all">總覽（所有時間）</option>
                {yearOptions.slice(1).map((year) => (
                  <option key={year} value={year}>
                    {year} 年
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-bold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 shadow-lg"
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                生成中...
              </>
            ) : (
              <>
                <Play className="w-6 h-6" />
                生成我的 Recap
              </>
            )}
          </button>
        </form>
      )}

      {error && (
        <div className="text-center text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800 max-w-3xl mx-auto">
          {error}
        </div>
      )}

      {/* Detailed View (after animation) */}
      {animationComplete && recapData && (
        <div className="space-y-8">
          {/* Replay Button */}
          <div className="text-center">
            <button
              onClick={replayAnimation}
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-bold text-lg transition-all flex items-center justify-center gap-3 shadow-lg mx-auto"
            >
              <Play className="w-6 h-6" />
              重播動畫
            </button>
          </div>

          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-8 rounded-xl text-center">
            <h2 className="text-4xl font-bold mb-2">
              {recapData.is_all_time ? "總覽 Recap" : `${recapData.year} Recap`}
            </h2>
            <p className="text-xl text-purple-100">{recapData.username}</p>
          </div>

          {/* Overview - All Top Rankings */}
          <div className="bg-gray-800 p-8 rounded-xl border border-gray-700">
            <h3 className="text-3xl font-bold mb-6 text-center bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
              你的年度回顧總覽
            </h3>
            <div className="grid grid-cols-3 md:grid-cols-3 lg:grid-cols-3 gap-2">
              {/* Top Anime */}
              {recapData.top_anime.length > 0 && (
                <div className="bg-gradient-to-br from-purple-600 to-purple-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">👑</div>
                  <h4 className="text-lg font-bold mb-2">最愛作品</h4>
                  <p className="text-xl text-purple-100 line-clamp-2">
                    {recapData.top_anime[0].title_english ||
                      recapData.top_anime[0].title}
                  </p>
                  <p className="text-2xl font-bold text-yellow-300 mt-2">
                    ⭐ {recapData.top_anime[0].score}
                  </p>
                </div>
              )}

              {/* Top Genre */}
              {Object.keys(recapData.genre_distribution).length > 0 && (
                <div className="bg-gradient-to-br from-pink-600 to-rose-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">🎭</div>
                  <h4 className="text-lg font-bold mb-2">最愛類型</h4>
                  <p className="text-xl text-pink-100 font-bold">
                    {Object.keys(recapData.genre_distribution)[0]}
                  </p>
                  <p className="text-base text-pink-200 mt-2">
                    {Object.values(recapData.genre_distribution)[0]} 部作品
                  </p>
                </div>
              )}

              {/* Top Studio */}
              {Object.keys(recapData.studio_distribution).length > 0 && (
                <div className="bg-gradient-to-br from-blue-600 to-blue-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">🏢</div>
                  <h4 className="text-lg font-bold mb-2">最愛製作公司</h4>
                  <p className="text-xl text-blue-100 line-clamp-2">
                    {Object.values(recapData.studio_distribution)[0].name}
                  </p>
                  <p className="text-base text-blue-200 mt-2">
                    {Object.values(recapData.studio_distribution)[0].count}{" "}
                    部作品
                  </p>
                </div>
              )}

              {/* Top Voice Actor */}
              {Object.keys(recapData.voice_actor_distribution).length > 0 && (
                <div className="bg-gradient-to-br from-cyan-600 to-cyan-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">🎤</div>
                  <h4 className="text-xl font-bold mb-2">最愛聲優</h4>
                  <p className="text-base text-cyan-100 line-clamp-2">
                    {Object.values(recapData.voice_actor_distribution)[0].name}
                  </p>
                  <p className="text-sm text-cyan-200 mt-1">
                    {
                      Object.values(recapData.voice_actor_distribution)[0]
                        .native
                    }
                  </p>
                </div>
              )}

              {/* Top Tag */}
              {Object.keys(recapData.tag_distribution).length > 0 && (
                <div className="bg-gradient-to-br from-green-600 to-green-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">🏷️</div>
                  <h4 className="text-lg font-bold mb-2">最感興趣的主題</h4>
                  <p className="text-xl text-green-100 font-bold">
                    {Object.keys(recapData.tag_distribution)[0]}
                  </p>
                  <p className="text-base text-green-200 mt-2">
                    {Object.values(recapData.tag_distribution)[0]} 部作品
                  </p>
                </div>
              )}

              {/* Top Season */}
              {Object.keys(recapData.season_distribution).length > 0 && (
                <div className="bg-gradient-to-br from-orange-600 to-orange-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">📅</div>
                  <h4 className="text-lg font-bold mb-2">最熱血的季節</h4>
                  <p className="text-xl text-orange-100">
                    {Object.keys(recapData.season_distribution)[0]}
                  </p>
                  <p className="text-xl font-bold text-orange-200 mt-2">
                    {Object.values(recapData.season_distribution)[0]} 部
                  </p>
                </div>
              )}

              {/* Most Rewatched */}
              {recapData.most_rewatched.length > 0 && (
                <div className="bg-gradient-to-br from-violet-600 to-violet-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">🔄</div>
                  <h4 className="text-lg font-bold mb-2">重看冠軍</h4>
                  <p className="text-xl text-violet-100 line-clamp-2">
                    {recapData.most_rewatched[0].title_english ||
                      recapData.most_rewatched[0].title}
                  </p>
                  <p className="text-xl font-bold text-violet-200 mt-2">
                    ×{recapData.most_rewatched[0].repeat_count}
                  </p>
                </div>
              )}

              {/* Total Stats */}
              <div className="bg-gradient-to-br from-indigo-600 to-indigo-800 p-6 rounded-xl">
                <div className="text-4xl mb-3">📊</div>
                <h4 className="text-lg font-bold mb-2">總計統計</h4>
                <p className="text-xl text-indigo-100">
                  {recapData.total_anime} 部動漫
                </p>
                <p className="text-base text-indigo-100">
                  {recapData.total_episodes} 集
                </p>
                <p className="text-base text-indigo-100">
                  {recapData.total_hours} 小時
                </p>
              </div>

              {/* Top Achievement */}
              {recapData.achievements.length > 0 && (
                <div className="bg-gradient-to-br from-yellow-600 to-yellow-800 p-6 rounded-xl">
                  <div className="text-4xl mb-3">
                    {recapData.achievements[0].icon}
                  </div>
                  <h4 className="text-lg font-bold mb-2">頂級成就</h4>
                  <p className="text-xl text-yellow-100 font-bold line-clamp-2">
                    {recapData.achievements[0].title}
                  </p>
                  <p className="text-sm text-yellow-200 mt-2 line-clamp-2">
                    {recapData.achievements[0].description}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <div className="flex items-center gap-3 mb-2">
                <Eye className="w-6 h-6 text-purple-400" />
                <span className="text-gray-400">總動漫數</span>
              </div>
              <div className="text-3xl font-bold">{recapData.total_anime}</div>
            </div>

            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <div className="flex items-center gap-3 mb-2">
                <BarChart3 className="w-6 h-6 text-blue-400" />
                <span className="text-gray-400">總集數</span>
              </div>
              <div className="text-3xl font-bold">
                {recapData.total_episodes}
              </div>
            </div>

            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <div className="flex items-center gap-3 mb-2">
                <Clock className="w-6 h-6 text-green-400" />
                <span className="text-gray-400">總時長</span>
              </div>
              <div className="text-3xl font-bold">{recapData.total_hours}h</div>
            </div>

            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <div className="flex items-center gap-3 mb-2">
                <Trophy className="w-6 h-6 text-yellow-400" />
                <span className="text-gray-400">完成數</span>
              </div>
              <div className="text-3xl font-bold">
                {recapData.completed_count}
              </div>
            </div>
          </div>

          {/* Status Breakdown */}
          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
            <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-purple-400" />
              狀態分布
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {recapData.completed_count}
                </div>
                <div className="text-sm text-gray-400">完成</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {recapData.watching_count}
                </div>
                <div className="text-sm text-gray-400">觀看中</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-400">
                  {recapData.planned_count}
                </div>
                <div className="text-sm text-gray-400">計劃中</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400">
                  {recapData.paused_count}
                </div>
                <div className="text-sm text-gray-400">暫停</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-400">
                  {recapData.dropped_count}
                </div>
                <div className="text-sm text-gray-400">棄番</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">
                  {recapData.repeating_count}
                </div>
                <div className="text-sm text-gray-400">重看</div>
              </div>
            </div>
          </div>

          {/* Top Anime */}
          {recapData.top_anime.length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Star className="w-6 h-6 text-yellow-400" />
                你的最愛 Top 10
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {recapData.top_anime.map((anime, index) => (
                  <div
                    key={anime.id}
                    className="relative group cursor-pointer hover:transform hover:scale-105 transition-all"
                    onClick={() =>
                      window.open(
                        `https://anilist.co/anime/${anime.id}`,
                        "_blank",
                      )
                    }
                  >
                    <div className="absolute -top-2 -left-2 bg-gradient-to-br from-yellow-400 to-yellow-600 text-black font-bold text-lg w-8 h-8 rounded-full flex items-center justify-center z-10 shadow-lg">
                      {index + 1}
                    </div>
                    <img
                      src={anime.coverImage}
                      alt={anime.title}
                      className="w-full h-48 object-cover rounded-lg shadow-lg"
                    />
                    <div className="mt-2">
                      <div className="text-sm font-semibold line-clamp-2">
                        {anime.title_english || anime.title}
                      </div>
                      <div className="flex items-center gap-1 mt-1">
                        <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                        <span className="text-yellow-400 font-bold text-sm">
                          {anime.score}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Genre Distribution */}
          {Object.keys(recapData.genre_distribution).length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-purple-400" />
                類型分布
              </h3>
              <div className="space-y-3">
                {Object.entries(recapData.genre_distribution)
                  .slice(0, 10)
                  .map(([genre, count], index) => {
                    const maxCount = Object.values(
                      recapData.genre_distribution,
                    )[0];
                    const percentage = (count / maxCount) * 100;
                    return (
                      <div key={genre}>
                        <div className="flex justify-between mb-1">
                          <span className="font-medium text-purple-300">
                            {genre}
                          </span>
                          <span className="text-gray-400">{count} 部</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full transition-all duration-1000"
                            style={{
                              width: `${percentage}%`,
                              animationDelay: `${index * 0.1}s`,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Achievements */}
          {recapData.achievements.length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Trophy className="w-6 h-6 text-yellow-400" />
                成就徽章
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recapData.achievements.map((achievement) => {
                  const tierColors = {
                    bronze: "from-orange-700 to-orange-900 border-orange-500",
                    silver: "from-gray-400 to-gray-600 border-gray-300",
                    gold: "from-yellow-500 to-yellow-700 border-yellow-400",
                    diamond: "from-cyan-400 to-blue-600 border-cyan-300",
                  };
                  const tierColor =
                    tierColors[achievement.tier] || tierColors.bronze;

                  return (
                    <div
                      key={achievement.id}
                      className={`bg-gradient-to-br ${tierColor} p-6 rounded-xl shadow-lg border-2`}
                    >
                      <div className="text-4xl mb-3">{achievement.icon}</div>
                      <h4 className="text-xl font-bold mb-2">
                        {achievement.title}
                      </h4>
                      <p className="text-white opacity-90">
                        {achievement.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tags Distribution */}
          {Object.keys(recapData.tag_distribution).length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-blue-400" />
                感興趣的主題 Top 10
              </h3>
              <div className="space-y-3">
                {Object.entries(recapData.tag_distribution)
                  .slice(0, 10)
                  .map(([tag, count], index) => {
                    const maxCount = Object.values(
                      recapData.tag_distribution,
                    )[0];
                    const percentage = (count / maxCount) * 100;
                    return (
                      <div key={tag}>
                        <div className="flex justify-between mb-1">
                          <span className="font-medium text-blue-300">
                            {tag}
                          </span>
                          <span className="text-gray-400">{count} 部</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-cyan-500 h-full rounded-full transition-all duration-1000"
                            style={{
                              width: `${percentage}%`,
                              animationDelay: `${index * 0.1}s`,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Studios Distribution */}
          {Object.keys(recapData.studio_distribution).length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Star className="w-6 h-6 text-pink-400" />
                最愛製作公司 Top 10
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.values(recapData.studio_distribution)
                  .slice(0, 10)
                  .map((studio, index) => (
                    <div
                      key={studio.id}
                      className="bg-gray-900 p-4 rounded-lg border border-pink-800 hover:border-pink-600 transition-all cursor-pointer"
                      onClick={() =>
                        studio.siteUrl && window.open(studio.siteUrl, "_blank")
                      }
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-pink-300">
                          #{index + 1}
                        </span>
                        <span className="text-gray-400">{studio.count} 部</span>
                      </div>
                      <div className="text-white font-semibold text-lg">
                        {studio.name}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Voice Actors Distribution */}
          {Object.keys(recapData.voice_actor_distribution).length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Star className="w-6 h-6 text-blue-400" />
                最常聽的聲優 Top 20
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {Object.values(recapData.voice_actor_distribution)
                  .slice(0, 20)
                  .map((va, index) => (
                    <div
                      key={va.id}
                      className="bg-gray-900 p-3 rounded-lg border border-blue-800 hover:border-blue-600 transition-all cursor-pointer group"
                      onClick={() =>
                        va.siteUrl && window.open(va.siteUrl, "_blank")
                      }
                    >
                      <div className="relative mb-2">
                        <div className="absolute -top-2 -left-2 bg-blue-600 text-white font-bold text-xs px-2 py-1 rounded-full z-10">
                          #{index + 1}
                        </div>
                        {va.image ? (
                          <img
                            src={va.image}
                            alt={va.name}
                            className="w-full h-32 object-cover rounded-lg group-hover:scale-105 transition-transform"
                          />
                        ) : (
                          <div className="w-full h-32 bg-gray-700 rounded-lg flex items-center justify-center">
                            <span className="text-4xl">👤</span>
                          </div>
                        )}
                      </div>
                      <div className="text-sm font-semibold text-white line-clamp-2 mb-1">
                        {va.name}
                      </div>
                      {va.native && (
                        <div className="text-xs text-gray-400 line-clamp-1 mb-1">
                          {va.native}
                        </div>
                      )}
                      <div className="text-xs text-blue-300">
                        {va.count} 部作品
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Season Distribution */}
          {Object.keys(recapData.season_distribution).length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Calendar className="w-6 h-6 text-green-400" />
                季節追番分布
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(recapData.season_distribution)
                  .slice(0, 10)
                  .map(([season, count]) => (
                    <div
                      key={season}
                      className="bg-gray-900 p-4 rounded-lg border border-green-800 flex items-center justify-between"
                    >
                      <span className="font-medium text-green-300">
                        {season}
                      </span>
                      <span className="text-2xl font-bold text-white">
                        {count}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Most Rewatched */}
          {recapData.most_rewatched.length > 0 && (
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Trophy className="w-6 h-6 text-purple-400" />
                重看次數最多
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {recapData.most_rewatched.map((anime) => (
                  <div
                    key={anime.id}
                    className="relative group cursor-pointer hover:transform hover:scale-105 transition-all"
                    onClick={() =>
                      window.open(
                        `https://anilist.co/anime/${anime.id}`,
                        "_blank",
                      )
                    }
                  >
                    <div className="absolute -top-2 -left-2 bg-purple-600 text-white font-bold text-sm px-2 py-1 rounded-full z-10 shadow-lg">
                      ×{anime.repeat_count}
                    </div>
                    <img
                      src={anime.coverImage}
                      alt={anime.title}
                      className="w-full h-48 object-cover rounded-lg shadow-lg"
                    />
                    <div className="mt-2">
                      <div className="text-sm font-semibold line-clamp-2">
                        {anime.title_english || anime.title}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Monthly Representative (Year mode only) */}
          {!recapData.is_all_time &&
            Object.keys(recapData.monthly_representative).length > 0 && (
              <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
                <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Calendar className="w-6 h-6 text-orange-400" />
                  {recapData.year} 年每月代表作
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {Object.entries(recapData.monthly_representative)
                    .sort(([a], [b]) => parseInt(a) - parseInt(b))
                    .map(([month, anime]) => {
                      const monthNames = [
                        "1月",
                        "2月",
                        "3月",
                        "4月",
                        "5月",
                        "6月",
                        "7月",
                        "8月",
                        "9月",
                        "10月",
                        "11月",
                        "12月",
                      ];
                      return (
                        <div
                          key={month}
                          className="relative group cursor-pointer hover:transform hover:scale-105 transition-all"
                          onClick={() =>
                            window.open(
                              `https://anilist.co/anime/${anime.id}`,
                              "_blank",
                            )
                          }
                        >
                          <div className="absolute -top-2 -left-2 bg-orange-600 text-white font-bold text-sm px-3 py-1 rounded-full z-10 shadow-lg">
                            {monthNames[anime.month - 1]}
                          </div>
                          <img
                            src={anime.coverImage}
                            alt={anime.title}
                            className="w-full h-48 object-cover rounded-lg shadow-lg"
                          />
                          <div className="mt-2">
                            <div className="text-sm font-semibold line-clamp-2">
                              {anime.title_english || anime.title}
                            </div>
                            <div className="flex items-center gap-1 mt-1">
                              <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                              <span className="text-yellow-400 font-bold text-sm">
                                {anime.score}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

          {/* New Recap Button */}
          <div className="text-center pt-8">
            <button
              onClick={() => {
                setRecapData(null);
                setAnimationComplete(false);
                setUsername("");
              }}
              className="px-8 py-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-bold text-lg transition-all"
            >
              生成新的 Recap
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
