import React from "react";

interface StatItemProps {
  icon: React.ReactNode;
  label: string;
  value: number | string;
}

/**
 * 統計項目組件
 * 用於顯示用戶的統計數據（如觀看數量、平均評分等）
 */
export const StatItem: React.FC<StatItemProps> = ({ icon, label, value }) => {
  return (
    <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
      <div className="flex items-center gap-3">
        <div className="text-gray-400">{icon}</div>
        <span className="text-gray-300">{label}</span>
      </div>
      <span className="text-xl font-bold">{value}</span>
    </div>
  );
};
