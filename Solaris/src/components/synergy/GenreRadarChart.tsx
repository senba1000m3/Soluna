import React from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { RadarData } from "../../types/synergy";

interface GenreRadarChartProps {
  radarData: RadarData;
  user1Name: string;
  user2Name: string;
}

/**
 * 類型偏好雷達圖組件
 *
 * 視覺化展示兩位用戶在不同動畫類型上的偏好強度
 *
 * 計算方式：
 * 1. 從用戶的動畫列表中提取類型偏好權重
 * 2. 權重計算考慮：用戶評分、觀看頻率
 * 3. 正規化到 0-100 範圍，便於比較
 * 4. 選取最重要的 8 個類型進行展示
 */
export const GenreRadarChart: React.FC<GenreRadarChartProps> = ({
  radarData,
  user1Name,
  user2Name,
}) => {
  // 將後端數據轉換為 recharts 所需格式
  const chartData = formatRadarData(radarData, user1Name, user2Name);

  return (
    <div className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={chartData}>
          {/* 網格線 */}
          <PolarGrid stroke="#374151" />

          {/* 角度軸（類型標籤） */}
          <PolarAngleAxis dataKey="subject" tick={{ fill: "#9CA3AF" }} />

          {/* 徑向軸（數值） */}
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: "#9CA3AF" }}
          />

          {/* 用戶1數據 */}
          <Radar
            name={user1Name}
            dataKey={user1Name}
            stroke="#3B82F6"
            fill="#3B82F6"
            fillOpacity={0.5}
          />

          {/* 用戶2數據 */}
          <Radar
            name={user2Name}
            dataKey={user2Name}
            stroke="#14B8A6"
            fill="#14B8A6"
            fillOpacity={0.5}
          />

          {/* 圖例 */}
          <Legend />

          {/* 提示框 */}
          <Tooltip
            contentStyle={{
              backgroundColor: "#1F2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

/**
 * 格式化雷達圖數據
 * 將後端返回的格式轉換為 recharts 所需的格式
 */
const formatRadarData = (
  radarData: RadarData,
  user1Name: string,
  user2Name: string
) => {
  return radarData.labels.map((label, idx) => ({
    subject: label,
    [user1Name]: radarData.user1[idx],
    [user2Name]: radarData.user2[idx],
  }));
};
