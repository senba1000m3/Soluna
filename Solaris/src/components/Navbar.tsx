import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Search, Sparkles, Users, Clock } from "lucide-react";

export const Navbar = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path
      ? "bg-purple-600 text-white"
      : "text-gray-300 hover:bg-gray-800 hover:text-white";
  };

  return (
    <nav className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center gap-2 group">
              <div className="bg-gradient-to-tr from-purple-500 to-pink-500 p-2 rounded-lg group-hover:scale-110 transition-transform duration-200">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
                Soluna
              </span>
            </Link>
          </div>

          <div className="flex space-x-2">
            <Link
              to="/"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/")}`}
            >
              <Search className="w-4 h-4" />
              搜尋
            </Link>

            <Link
              to="/recommend"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/recommend")}`}
            >
              <Sparkles className="w-4 h-4" />
              新番預測
            </Link>

            <Link
              to="/synergy"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/synergy")}`}
            >
              <Users className="w-4 h-4" />
              共鳴配對
            </Link>

            <Link
              to="/timeline"
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${isActive("/timeline")}`}
            >
              <Clock className="w-4 h-4" />
              動畫大世紀
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};
