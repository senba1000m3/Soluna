import React, { useState, useRef, useEffect } from "react";
import { AlertTriangle } from "lucide-react";
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
import { BACKEND_URL } from "../config/env";

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

  // Use ref to track current request
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentTaskIdRef = useRef<string>("");

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) {
      setError("請輸入使用者名稱");
      return;
    }

    // **Abort previous request if exists**
    if (abortControllerRef.current) {
      console.log("Aborting previous request...");
      abortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();

    // Clear previous results and errors before starting new analysis
    setLoading(true);
    setError("");
    setResult(null);

    console.log(`Starting analysis for user: ${username.trim()}`);

    try {
      // Send the analyze request (without task_id, no progress tracking)
      const response = await fetch(`${BACKEND_URL}/analyze_drops`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: username.trim(),
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "分析請求失敗，請確認 ID 是否正確");
      }

      const data: AnalyzeResponse = await response.json();
      setResult(data);
      console.log(`Analysis completed successfully`);
    } catch (err: any) {
      // Ignore abort errors
      if (err.name === "AbortError") {
        console.log("Request was aborted");
        return;
      }

      console.error(err);
      setError(err.message || "發生錯誤，請稍後再試");
    } finally {
      setLoading(false);
    }
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

      {error && (
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
