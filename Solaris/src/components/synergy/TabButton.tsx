import React from "react";

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

/**
 * 標籤按鈕組件
 * 用於共鳴配對頁面的標籤切換導航
 */
export const TabButton: React.FC<TabButtonProps> = ({
  active,
  onClick,
  icon,
  label,
}) => {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap ${
        active
          ? "bg-gradient-to-r from-blue-600 to-teal-600 text-white"
          : "bg-gray-700 text-gray-300 hover:bg-gray-600"
      }`}
    >
      {icon}
      {label}
    </button>
  );
};
