import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Sparkles,
  Users,
  Clock,
  AlertTriangle,
  Film,
  User,
  Star,
  ChevronDown,
  Plus,
  X,
  Search as SearchIcon,
} from "lucide-react";
import { useGlobalUser } from "../contexts/AuthContext";

export const Navbar = () => {
  const location = useLocation();
  const { mainUser, quickIds, loginUser, logoutUser } = useGlobalUser();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showQuickIdMenu, setShowQuickIdMenu] = useState(false);
  const [showSetUserModal, setShowSetUserModal] = useState(false);
  const [tempUsername, setTempUsername] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const isActive = (path: string) => {
    return location.pathname === path
      ? "bg-purple-600 text-white"
      : "text-gray-300 hover:bg-gray-800 hover:text-white";
  };

  const handleSetUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      await loginUser(tempUsername);
      setShowSetUserModal(false);
      setTempUsername("");
    } catch (err: any) {
      setError(err.message || "找不到此使用者，請檢查使用者名稱");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logoutUser();
      setShowUserMenu(false);
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <>
      <nav className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50 shadow-lg py-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* 左側：Logo + 功能選單 */}
            <div className="flex items-center gap-4">
              {/* Logo */}
              <Link to="/" className="flex items-center gap-2 group">
                <div className="bg-gradient-to-tr from-purple-500 to-pink-500 p-2 rounded-lg group-hover:scale-110 transition-transform duration-200">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
                  Soluna
                </span>
              </Link>

              {/* 功能選單 */}
              <div className="hidden md:flex items-center gap-1">
                <Link
                  to="/recommend"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/recommend")}`}
                >
                  <Sparkles className="w-4 h-4" />
                  新番預測
                </Link>

                <Link
                  to="/synergy"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/synergy")}`}
                >
                  <Users className="w-4 h-4" />
                  共鳴配對
                </Link>

                <Link
                  to="/timeline"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/timeline")}`}
                >
                  <Clock className="w-4 h-4" />
                  動畫大世紀
                </Link>

                <Link
                  to="/drop-predict"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/drop-predict")}`}
                >
                  <AlertTriangle className="w-4 h-4" />
                  棄番預測
                </Link>

                <Link
                  to="/recap"
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/recap")}`}
                >
                  <Film className="w-4 h-4" />
                  動態回顧
                </Link>
              </div>
            </div>

            {/* 右側：快速 ID + 搜尋 + 使用者 */}
            <div className="flex items-center gap-3">
              {/* 快速 ID 選單 */}
              {quickIds.length > 0 && (
                <div className="relative">
                  <button
                    onClick={() => setShowQuickIdMenu(!showQuickIdMenu)}
                    className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  >
                    <Star className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm">快速 ID</span>
                    <ChevronDown className="w-4 h-4" />
                  </button>

                  {showQuickIdMenu && (
                    <div className="absolute right-0 mt-2 w-64 bg-gray-800 rounded-lg shadow-xl border border-gray-700 py-2 max-h-80 overflow-y-auto">
                      <div className="px-3 py-2 text-xs text-gray-400 border-b border-gray-700">
                        快速切換
                      </div>
                      {quickIds.map((qid) => (
                        <button
                          key={qid.id}
                          className="w-full px-3 py-2 text-left hover:bg-gray-700 flex items-center gap-3 transition-colors"
                          onClick={() => {
                            setTempUsername(qid.anilistUsername);
                            setShowQuickIdMenu(false);
                            setShowSetUserModal(true);
                          }}
                        >
                          <img
                            src={qid.avatar}
                            alt={qid.anilistUsername}
                            className="w-8 h-8 rounded-full object-cover"
                          />
                          <div className="flex-1">
                            <div className="text-sm font-medium text-white">
                              {qid.nickname || qid.anilistUsername}
                            </div>
                            <div className="text-xs text-gray-400">
                              @{qid.anilistUsername}
                            </div>
                          </div>
                        </button>
                      ))}
                      <div className="border-t border-gray-700 mt-2 pt-2">
                        <Link
                          to="/settings/quick-ids"
                          className="w-full px-3 py-2 text-sm text-purple-400 hover:bg-gray-700 flex items-center gap-2 transition-colors"
                          onClick={() => setShowQuickIdMenu(false)}
                        >
                          <Plus className="w-4 h-4" />
                          管理快速 ID
                        </Link>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 搜尋按鈕 */}
              <Link
                to="/"
                className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/")}`}
              >
                <SearchIcon className="w-4 h-4" />
                <span className="hidden sm:inline">搜尋</span>
              </Link>

              {/* 使用者按鈕 */}
              <div className="relative">
                {mainUser ? (
                  <>
                    <button
                      onClick={() => setShowUserMenu(!showUserMenu)}
                      className="flex items-center gap-2 px-2 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                    >
                      <img
                        src={mainUser.avatar}
                        alt={mainUser.anilistUsername}
                        className="w-8 h-8 rounded-full object-cover border-2 border-purple-400"
                      />
                      <span className="text-sm font-medium hidden sm:inline">
                        {mainUser.anilistUsername}
                      </span>
                      <ChevronDown className="w-4 h-4 hidden sm:inline" />
                    </button>

                    {showUserMenu && (
                      <div className="absolute right-0 mt-2 w-56 bg-gray-800 rounded-lg shadow-xl border border-gray-700 py-2">
                        <div className="px-4 py-3 border-b border-gray-700">
                          <div className="flex items-center gap-3">
                            <img
                              src={mainUser.avatar}
                              alt={mainUser.anilistUsername}
                              className="w-10 h-10 rounded-full object-cover"
                            />
                            <div>
                              <div className="font-medium text-white">
                                {mainUser.anilistUsername}
                              </div>
                              <div className="text-xs text-gray-400">
                                ID: {mainUser.anilistId}
                              </div>
                            </div>
                          </div>
                        </div>
                        <Link
                          to="/settings/quick-ids"
                          className="w-full px-4 py-2 text-left hover:bg-gray-700 flex items-center gap-2 text-sm transition-colors"
                          onClick={() => setShowUserMenu(false)}
                        >
                          <Plus className="w-4 h-4" />
                          管理快速 ID
                        </Link>
                        <button
                          onClick={handleLogout}
                          className="w-full px-4 py-2 text-left hover:bg-gray-700 flex items-center gap-2 text-sm text-red-400 transition-colors"
                        >
                          <X className="w-4 h-4" />
                          登出
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  <button
                    onClick={() => setShowSetUserModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                  >
                    <User className="w-4 h-4" />
                    <span className="text-sm font-medium hidden sm:inline">
                      設定使用者
                    </span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* 設定使用者 Modal */}
      {showSetUserModal && (
        <div
          className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4"
          onClick={() => setShowSetUserModal(false)}
        >
          <div
            className="bg-gray-800 rounded-xl max-w-md w-full p-6 border border-gray-700 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
                設定全局使用者
              </h2>
              <button
                onClick={() => setShowSetUserModal(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="mb-4 p-3 bg-blue-900/20 border border-blue-700 rounded-lg">
              <p className="text-sm text-blue-300">
                💡 輸入你的 AniList 使用者名稱，將自動套用到所有頁面
              </p>
            </div>

            <form onSubmit={handleSetUser} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-300">
                  AniList 使用者名稱
                </label>
                <input
                  type="text"
                  value={tempUsername}
                  onChange={(e) => setTempUsername(e.target.value)}
                  required
                  className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
                  placeholder="例如: senba1000m3"
                />
                <p className="text-xs text-gray-500 mt-1">
                  系統會自動抓取你的頭像和 ID
                </p>
              </div>

              {error && (
                <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded-lg border border-red-800">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    載入中...
                  </>
                ) : (
                  <>
                    <User className="w-4 h-4" />
                    確認設定
                  </>
                )}
              </button>
            </form>

            <div className="mt-4 pt-4 border-t border-gray-700">
              <p className="text-xs text-gray-500 mb-2">
                快速選擇（如果你已經新增過）
              </p>
              {quickIds.length > 0 ? (
                <div className="space-y-2">
                  {quickIds.slice(0, 3).map((qid) => (
                    <button
                      key={qid.id}
                      onClick={() => {
                        setTempUsername(qid.anilistUsername);
                      }}
                      className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors text-left"
                    >
                      <img
                        src={qid.avatar}
                        alt={qid.anilistUsername}
                        className="w-6 h-6 rounded-full object-cover"
                      />
                      <span className="text-sm text-white">
                        {qid.nickname || qid.anilistUsername}
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <Link
                  to="/settings/quick-ids"
                  onClick={() => setShowSetUserModal(false)}
                  className="text-sm text-purple-400 hover:text-purple-300 transition-colors"
                >
                  前往新增常用的 ID →
                </Link>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};
