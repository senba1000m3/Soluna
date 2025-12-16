import React, { useState } from "react";
import { AlertTriangle } from "lucide-react";
import { ProgressMonitor } from "../components/ProgressMonitor";
import {
  AnalysisForm,
  AnimeDetailModal,
  ModelMetricsCard,
  DropPatternStats,
  AnimeListSection,
  AnimeItem,
  ModelMetrics,
  DropPatterns,
} from "../components/dropPredict";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface AnalyzeResponse {
  task_id: string;
  username: string;
  dropped_count: number;
  dropped_list: AnimeItem[];
  watching_list: AnimeItem[];
  planning_list: AnimeItem[];
  model_metrics: ModelMetrics;
  drop_patterns: DropPatterns;
}

export const DropPredict = () => {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState("");
  const [modalAnime, setModalAnime] = useState<AnimeItem | null>(null);
  const [taskId, setTaskId] = useState<string>("");
  const [showProgress, setShowProgress] = useState(false);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) {
      setError("請輸入使用者名稱");
      return;
    }

    // Clear previous results and errors before starting new analysis
    setLoading(true);
    setError("");
    setResult(null);
    setShowProgress(true);

    // Generate a task ID
    const newTaskId = `drop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setTaskId(newTaskId);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze_drops`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: username.trim(),
          task_id: newTaskId,
        }),
      });

      if (!response.ok) {
        throw new Error("分析請求失敗，請確認 ID 是否正確");
      }

      const data: AnalyzeResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "發生錯誤，請稍後再試");
      setShowProgress(false);
    } finally {
      setLoading(false);
    }
  };

  const handleProgressComplete = () => {
    console.log("Progress completed");
    setShowProgress(false);
  };

  const handleProgressError = (error: string) => {
    console.error("Progress error:", error);
    setError(error);
    setShowProgress(false);
    setLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto px-4">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-4 mt-5 bg-gradient-to-r from-red-500 to-orange-500 text-transparent bg-clip-text flex items-center justify-center gap-3">
          <AlertTriangle className="w-10 h-10 text-red-500" />
          棄番風險預測
        </h1>
        <p className="text-gray-400">
          分析你的棄番歷史，訓練 AI 模型，找出你的「棄番地雷區」
        </p>
      </div>

      <AnalysisForm
        username={username}
        onUsernameChange={setUsername}
        onSubmit={handleAnalyze}
        loading={loading}
      />

      {/* Progress Monitor */}
      {showProgress && taskId && (
        <div className="mb-8 max-w-2xl mx-auto">
          <ProgressMonitor
            taskId={taskId}
            onComplete={handleProgressComplete}
            onError={handleProgressError}
          />
        </div>
      )}

      {error && !showProgress && (
        <div className="text-center text-red-400 mb-8 bg-red-900/20 p-4 rounded-lg border border-red-800 max-w-2xl mx-auto">
          {error}
        </div>
      )}

      {/* Modal for showing prediction reasons */}
      <AnimeDetailModal
        anime={modalAnime}
        onClose={() => setModalAnime(null)}
      />

      {result && (
        <div className="space-y-12 animate-fade-in">
          {/* Watching List with Predictions */}
          <AnimeListSection
            title={`正在觀看 - 棄番風險預測 (${result.watching_list.filter((a) => (a.drop_probability ?? 0) > 0).length} 部)`}
            animeList={result.watching_list}
            variant="watching"
            icon="warning"
            iconColor="text-yellow-500"
            onShowDetails={setModalAnime}
          />

          {/* Planning List with Predictions */}
          <AnimeListSection
            title={`預定觀看 - 棄番風險預測 (${result.planning_list.filter((a) => (a.drop_probability ?? 0) > 0).length} 部)`}
            animeList={result.planning_list}
            variant="planning"
            icon="warning"
            iconColor="text-purple-500"
            onShowDetails={setModalAnime}
            limit={6}
          />

          {/* Model Metrics Section */}
          <ModelMetricsCard metrics={result.model_metrics} />

          {/* Drop Patterns Statistics */}
          {result.drop_patterns && (
            <DropPatternStats patterns={result.drop_patterns} />
          )}

          {/* Dropped List Section */}
          <AnimeListSection
            title={`你的棄番黑歷史 (${result.dropped_count} 部)`}
            animeList={result.dropped_list}
            variant="dropped"
            icon="x"
            iconColor="text-red-500"
            emptyMessage="太強了！你沒有棄番紀錄"
            emptySubMessage="你是個有始有終的觀眾，或者你還沒開始整理清單？"
          />
        </div>
      )}
    </div>
  );
};
