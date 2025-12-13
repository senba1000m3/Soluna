import React, { useState } from "react";
import { Users, Heart, Zap, Loader2, AlertCircle } from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface UserProfile {
  name: string;
  avatar: {
    large: string;
  };
}

interface CommonGenre {
  genre: string;
  score: number;
}

interface SynergyResponse {
  user1: UserProfile;
  user2: UserProfile;
  compatibility_score: number;
  common_genres: CommonGenre[];
  message: string;
}

export const Synergy = () => {
  const [user1, setUser1] = useState("");
  const [user2, setUser2] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<SynergyResponse | null>(null);

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user1.trim() || !user2.trim()) {
      setError("請輸入兩個使用者名稱");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${BACKEND_URL}/pair_compare`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user1: user1.trim(),
          user2: user2.trim(),
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "比較請求失敗");
      }

      const data: SynergyResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "發生錯誤，請確認使用者名稱是否正確");
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-400";
    if (score >= 60) return "text-yellow-400";
    return "text-red-400";
  };

  const getScoreMessage = (score: number) => {
    if (score >= 90) return "靈魂伴侶！你們的品味驚人地相似！";
    if (score >= 80) return "非常合拍！有很多共同話題。";
    if (score >= 60) return "還不錯，有些共同喜好。";
    if (score >= 40) return "品味有些差異，但可以互相推坑。";
    return "水火不容？或許是互補的關係！";
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-teal-400 text-transparent bg-clip-text flex items-center justify-center gap-3">
          <Users className="w-10 h-10 text-blue-400" />
          共鳴配對
        </h1>
        <p className="text-gray-400">
          輸入兩個 AniList ID，分析你們的動畫品味契合度
        </p>
      </div>

      <form
        onSubmit={handleCompare}
        className="mb-12 bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-xl"
      >
        <div className="flex flex-col md:flex-row gap-6 items-center justify-center mb-8">
          <div className="w-full">
            <label className="block text-sm font-medium mb-2 text-gray-300">
              使用者 A
            </label>
            <input
              type="text"
              value={user1}
              onChange={(e) => setUser1(e.target.value)}
              placeholder="例如: Gigguk"
              className="w-full px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-lg"
            />
          </div>

          <div className="flex items-center justify-center pt-6">
            <div className="bg-gray-700 p-3 rounded-full">
              <Zap className="w-6 h-6 text-yellow-400 fill-yellow-400" />
            </div>
          </div>

          <div className="w-full">
            <label className="block text-sm font-medium mb-2 text-gray-300">
              使用者 B
            </label>
            <input
              type="text"
              value={user2}
              onChange={(e) => setUser2(e.target.value)}
              placeholder="例如: TrashTaste"
              className="w-full px-4 py-3 bg-gray-700 rounded-lg border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-lg"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 rounded-lg font-bold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
        >
          {loading ? (
            <>
              <Loader2 className="w-6 h-6 animate-spin" />
              正在分析共鳴...
            </>
          ) : (
            <>
              <Heart className="w-6 h-6" />
              開始配對分析
            </>
          )}
        </button>
      </form>

      {error && (
        <div className="flex items-center justify-center gap-2 text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {result && (
        <div className="animate-fade-in space-y-8">
          {/* Profiles & Score */}
          <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 text-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-blue-500 to-teal-500" />

            <div className="flex flex-col md:flex-row items-center justify-center gap-8 mb-8">
              <div className="flex flex-col items-center">
                <img
                  src={result.user1.avatar.large}
                  alt={result.user1.name}
                  className="w-24 h-24 rounded-full border-4 border-blue-500 shadow-lg mb-3"
                />
                <h3 className="text-xl font-bold">{result.user1.name}</h3>
              </div>

              <div className="flex flex-col items-center justify-center px-8">
                <div className="text-6xl font-black mb-2 flex items-baseline gap-1">
                  <span className={getScoreColor(result.compatibility_score)}>
                    {result.compatibility_score.toFixed(1)}
                  </span>
                  <span className="text-2xl text-gray-500">%</span>
                </div>
                <div className="text-sm text-gray-400 uppercase tracking-widest font-semibold">
                  共鳴指數
                </div>
              </div>

              <div className="flex flex-col items-center">
                <img
                  src={result.user2.avatar.large}
                  alt={result.user2.name}
                  className="w-24 h-24 rounded-full border-4 border-teal-500 shadow-lg mb-3"
                />
                <h3 className="text-xl font-bold">{result.user2.name}</h3>
              </div>
            </div>

            <p className="text-xl text-blue-200 font-medium bg-blue-900/30 py-3 px-6 rounded-full inline-block">
              {getScoreMessage(result.compatibility_score)}
            </p>
          </div>

          {/* Common Genres */}
          {result.common_genres.length > 0 && (
            <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Zap className="w-6 h-6 text-yellow-400" />
                共同喜好領域
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.common_genres.map((genre, idx) => (
                  <div
                    key={genre.genre}
                    className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-between"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-gray-400 font-mono text-lg">
                        #{idx + 1}
                      </span>
                      <span className="font-bold text-lg">{genre.genre}</span>
                    </div>
                    <div className="h-2 w-24 bg-gray-600 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-400 to-teal-400"
                        style={{ width: "100%" }} // Simplified visual
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
