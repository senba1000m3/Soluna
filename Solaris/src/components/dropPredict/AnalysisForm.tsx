import React from "react";
import { Loader2, BrainCircuit } from "lucide-react";
import { QuickIDSelector } from "../QuickIDSelector";

interface AnalysisFormProps {
  username: string;
  onUsernameChange: (username: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
}

export const AnalysisForm: React.FC<AnalysisFormProps> = ({
  username,
  onUsernameChange,
  onSubmit,
  loading,
}) => {
  return (
    <form
      onSubmit={onSubmit}
      className="mb-12 bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-xl max-w-2xl mx-auto"
    >
      <div className="space-y-4">
        <div className="flex flex-col md:flex-row gap-4">
          <QuickIDSelector
            value={username}
            onChange={onUsernameChange}
            label="AniList 使用者名稱"
            placeholder="輸入 AniList ID（例如：senba1000m3）"
            required={true}
            className="flex-1"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-bold text-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                訓練模型中...
              </>
            ) : (
              <>
                <BrainCircuit className="w-6 h-6" />
                開始分析
              </>
            )}
          </button>
        </div>
        <p className="text-xs text-gray-400 bg-gray-900/50 p-3 rounded-lg text-center">
          💡 每次分析都會自動從 AniList 抓取最新資料，所以你在 AniList
          上的任何變更都會被反映。
          <br />
          首次分析或資料變更較多時，可能需要幾秒鐘的時間。
        </p>
      </div>
    </form>
  );
};
