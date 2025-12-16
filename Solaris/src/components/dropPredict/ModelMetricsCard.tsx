import React from "react";
import { BarChart2 } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export interface ModelMetrics {
  accuracy: number;
  sample_size: number;
  dropped_count: number;
  completed_count: number;
  top_features: [string, number][];
  error?: string;
}

interface ModelMetricsCardProps {
  metrics: ModelMetrics;
}

export const ModelMetricsCard: React.FC<ModelMetricsCardProps> = ({
  metrics,
}) => {
  return (
    <div className="bg-gray-800 rounded-xl p-8 border border-gray-700 shadow-lg">
      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
        <BarChart2 className="w-6 h-6 text-blue-400" />
        AI 模型訓練報告
      </h2>

      {metrics.error ? (
        <div className="text-yellow-400 bg-yellow-900/20 p-4 rounded-lg">
          ⚠️ {metrics.error}
          <p className="text-sm mt-1 text-gray-400">
            可能是因為你的棄番或完食紀錄太少，無法進行有效的機器學習訓練。
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Stats Cards */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-700/50 p-4 rounded-lg text-center">
              <p className="text-gray-400 text-sm mb-1">模型準確率</p>
              <p className="text-3xl font-bold text-green-400">
                {(metrics.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-gray-700/50 p-4 rounded-lg text-center">
              <p className="text-gray-400 text-sm mb-1">訓練樣本數</p>
              <p className="text-3xl font-bold text-blue-400">
                {metrics.sample_size}
              </p>
            </div>
            <div className="bg-gray-700/50 p-4 rounded-lg text-center">
              <p className="text-gray-400 text-sm mb-1">完食作品</p>
              <p className="text-3xl font-bold text-purple-400">
                {metrics.completed_count}
              </p>
            </div>
            <div className="bg-gray-700/50 p-4 rounded-lg text-center">
              <p className="text-gray-400 text-sm mb-1">棄番作品</p>
              <p className="text-3xl font-bold text-red-400">
                {metrics.dropped_count}
              </p>
            </div>
          </div>

          {/* Feature Importance Chart */}
          <div className="bg-gray-900/50 p-4 rounded-lg h-64">
            <p className="text-gray-400 text-sm mb-4 text-center">
              影響棄番的關鍵因素 (Top 5)
            </p>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={metrics.top_features
                  .slice(0, 5)
                  .map(([name, val]) => ({
                    name: name.replace("Genre_", "").replace("Tag_", ""),
                    value: val,
                  }))}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" hide />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={100}
                  tick={{ fill: "#9CA3AF", fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1F2937",
                    borderColor: "#374151",
                    color: "#F3F4F6",
                  }}
                />
                <Bar dataKey="value" fill="#F87171" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};
