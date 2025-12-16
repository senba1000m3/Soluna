import React, { useState } from "react";
import { useGlobalUser } from "../contexts/AuthContext";
import { Star, ChevronDown, User, Plus } from "lucide-react";
import { Link } from "react-router-dom";

interface QuickIDSelectorProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  placeholder?: string;
  required?: boolean;
  className?: string;
}

export const QuickIDSelector: React.FC<QuickIDSelectorProps> = ({
  value,
  onChange,
  label = "AniList 使用者名稱",
  placeholder = "輸入使用者名稱或從快速選單選擇",
  required = false,
  className = "",
}) => {
  const { mainUser, quickIds } = useGlobalUser();
  const [showDropdown, setShowDropdown] = useState(false);

  const handleQuickSelect = (username: string) => {
    onChange(username);
    setShowDropdown(false);
  };

  // 判斷是否有可用的快速選項（主 ID + 常用 ID）
  const hasQuickOptions = mainUser || quickIds.length > 0;

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-2">
        <label className="block text-sm font-medium text-gray-300">
          {label}
          {required && <span className="text-red-400 ml-1">*</span>}
        </label>
        {hasQuickOptions && (
          <button
            type="button"
            onClick={() => setShowDropdown(!showDropdown)}
            className="text-xs text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1"
          >
            <Star className="w-3 h-3" />
            快速選擇
            <ChevronDown className="w-3 h-3" />
          </button>
        )}
      </div>

      <div className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          required={required}
          placeholder={placeholder}
          className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
        />

        {/* 快速選擇下拉選單 */}
        {showDropdown && hasQuickOptions && (
          <div className="absolute z-10 w-full mt-2 bg-gray-800 rounded-lg shadow-xl border border-gray-700 py-2 max-h-80 overflow-y-auto">
            <div className="px-3 py-2 text-xs text-gray-400 border-b border-gray-700">
              快速選擇
            </div>

            {/* 主 ID（如果存在） */}
            {mainUser && (
              <>
                <div className="px-3 py-1 text-xs text-gray-500 mt-2">
                  主 ID
                </div>
                <button
                  type="button"
                  onClick={() => handleQuickSelect(mainUser.anilistUsername)}
                  className={`w-full px-3 py-2 text-left hover:bg-gray-700 flex items-center gap-3 transition-colors ${
                    value === mainUser.anilistUsername ? "bg-gray-700/50" : ""
                  }`}
                >
                  <img
                    src={mainUser.avatar}
                    alt={mainUser.anilistUsername}
                    className="w-8 h-8 rounded-full object-cover"
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white flex items-center gap-2">
                      {mainUser.anilistUsername}
                      <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                    </div>
                    <div className="text-xs text-gray-400">
                      ID: {mainUser.anilistId}
                    </div>
                  </div>
                  {value === mainUser.anilistUsername && (
                    <Star className="w-4 h-4 text-purple-400 fill-purple-400" />
                  )}
                </button>
              </>
            )}

            {/* 常用 ID 列表 */}
            {quickIds.length > 0 && (
              <>
                <div className="px-3 py-1 text-xs text-gray-500 mt-2 border-t border-gray-700 pt-2">
                  常用 ID
                </div>
                {quickIds.map((qid) => (
                  <button
                    key={qid.id}
                    type="button"
                    onClick={() => handleQuickSelect(qid.anilistUsername)}
                    className={`w-full px-3 py-2 text-left hover:bg-gray-700 flex items-center gap-3 transition-colors ${
                      value === qid.anilistUsername ? "bg-gray-700/50" : ""
                    }`}
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
                        @{qid.anilistUsername} • ID: {qid.anilistId}
                      </div>
                    </div>
                    {value === qid.anilistUsername && (
                      <Star className="w-4 h-4 text-purple-400 fill-purple-400" />
                    )}
                  </button>
                ))}
              </>
            )}

            <div className="border-t border-gray-700 mt-2 pt-2">
              <Link
                to="/settings/quick-ids"
                className="w-full px-3 py-2 text-sm text-purple-400 hover:bg-gray-700 flex items-center gap-2 transition-colors"
                onClick={() => setShowDropdown(false)}
              >
                <Plus className="w-4 h-4" />
                管理快速 ID
              </Link>
            </div>
          </div>
        )}
      </div>

      {/* 當前選擇提示 */}
      {value && hasQuickOptions && (
        <div className="mt-2 flex items-center gap-2">
          {mainUser && value === mainUser.anilistUsername ? (
            <span className="text-xs text-yellow-400 flex items-center gap-1">
              <Star className="w-3 h-3 fill-yellow-400" />主 ID
            </span>
          ) : quickIds.find((qid) => qid.anilistUsername === value) ? (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <Star className="w-3 h-3 fill-green-400" />
              常用 ID
            </span>
          ) : (
            <span className="text-xs text-gray-500">手動輸入</span>
          )}
        </div>
      )}

      {/* 未設定使用者時的提示 */}
      {!mainUser && quickIds.length === 0 && (
        <p className="text-xs text-gray-500 mt-1">
          點擊右上角
          <Link
            to="/"
            className="text-purple-400 hover:text-purple-300 mx-1"
            onClick={(e) => {
              e.preventDefault();
              // 觸發設定使用者 modal
              const setUserBtn = document.querySelector(
                '[aria-label="設定使用者"]',
              ) as HTMLButtonElement;
              if (setUserBtn) setUserBtn.click();
            }}
          >
            設定使用者
          </Link>
          以啟用快速選擇
        </p>
      )}
    </div>
  );
};
