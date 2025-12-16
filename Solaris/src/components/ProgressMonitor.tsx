import React, { useEffect, useState, useRef } from "react";
import { Loader2, CheckCircle2, XCircle, AlertCircle } from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

interface ProgressUpdate {
  task_id: string;
  progress: number;
  total: number;
  percentage: number;
  message: string;
  stage: string;
  status: "pending" | "running" | "completed" | "error";
  timestamp: number;
  heartbeat?: boolean;
}

interface ProgressMonitorProps {
  taskId: string;
  onComplete?: (data: ProgressUpdate) => void;
  onError?: (error: string) => void;
  autoClose?: boolean;
  autoCloseDelay?: number;
}

export const ProgressMonitor: React.FC<ProgressMonitorProps> = ({
  taskId,
  onComplete,
  onError,
  autoClose = false,
  autoCloseDelay = 2000,
}) => {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string>("");
  const eventSourceRef = useRef<EventSource | null>(null);
  const autoCloseTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!taskId) return;

    // Reset state for new task
    setProgress(null);
    setError("");
    setIsConnected(false);

    // Close previous connection if exists
    if (eventSourceRef.current) {
      console.log("Closing previous SSE connection");
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    console.log(`Opening SSE connection for task: ${taskId}`);

    // Connect to SSE endpoint
    const eventSource = new EventSource(`${BACKEND_URL}/progress/${taskId}`);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log("SSE connection opened for task:", taskId);
      setIsConnected(true);
    };

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressUpdate = JSON.parse(event.data);
        console.log("Progress update:", data);
        setProgress(data);

        // Handle completion
        if (data.status === "completed") {
          console.log("Task completed:", taskId);
          if (onComplete) {
            onComplete(data);
          }

          // Auto close after delay if enabled
          if (autoClose) {
            autoCloseTimerRef.current = setTimeout(() => {
              console.log("Auto-closing SSE connection");
              eventSource.close();
            }, autoCloseDelay);
          }
        }

        // Handle error
        if (data.status === "error") {
          console.error("Task error:", data.message);
          setError(data.message || "發生錯誤");
          if (onError) {
            onError(data.message || "Unknown error");
          }
        }
      } catch (err) {
        console.error("Error parsing SSE data:", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE error for task:", taskId, err);
      setIsConnected(false);

      // Only show connection error if we haven't received any progress yet
      if (!progress) {
        setError("連線失敗，請重試");
      }

      eventSource.close();
    };

    // Cleanup
    return () => {
      console.log("Cleaning up SSE connection for task:", taskId);
      if (autoCloseTimerRef.current) {
        clearTimeout(autoCloseTimerRef.current);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [taskId, onComplete, onError, autoClose, autoCloseDelay]);

  if (!progress) {
    return (
      <div className="flex items-center gap-3 p-4 bg-gray-800 rounded-lg border border-gray-700">
        <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
        <span className="text-gray-300">正在連接...</span>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (progress.status) {
      case "completed":
        return <CheckCircle2 className="w-6 h-6 text-green-400" />;
      case "error":
        return <XCircle className="w-6 h-6 text-red-400" />;
      case "running":
        return <Loader2 className="w-6 h-6 animate-spin text-purple-400" />;
      default:
        return <AlertCircle className="w-6 h-6 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (progress.status) {
      case "completed":
        return "border-green-700 bg-green-900/20";
      case "error":
        return "border-red-700 bg-red-900/20";
      case "running":
        return "border-purple-700 bg-purple-900/20";
      default:
        return "border-gray-700 bg-gray-800";
    }
  };

  const getProgressBarColor = () => {
    switch (progress.status) {
      case "completed":
        return "bg-gradient-to-r from-green-500 to-green-600";
      case "error":
        return "bg-gradient-to-r from-red-500 to-red-600";
      default:
        return "bg-gradient-to-r from-purple-500 to-pink-500";
    }
  };

  return (
    <div
      className={`p-6 rounded-xl border-2 transition-all duration-300 ${getStatusColor()}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {getStatusIcon()}
          <div>
            <h3 className="font-semibold text-lg text-white">
              {progress.stage ? getStageLabel(progress.stage) : "處理中"}
            </h3>
            <p className="text-sm text-gray-400">
              {isConnected ? "即時更新" : "連線中斷"}
            </p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-purple-300">
            {progress.percentage.toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">
            {progress.progress}/{progress.total}
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full ${getProgressBarColor()} transition-all duration-500 ease-out`}
            style={{ width: `${progress.percentage}%` }}
          />
        </div>
      </div>

      {/* Message */}
      <div className="text-sm text-gray-300">
        {progress.message || "處理中..."}
      </div>

      {/* Error Message */}
      {error && progress.status === "error" && (
        <div className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded-lg">
          <div className="flex items-center gap-2 text-red-300">
            <XCircle className="w-4 h-4" />
            <span className="font-medium">錯誤:</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Success Message */}
      {progress.status === "completed" && (
        <div className="mt-4 p-3 bg-green-900/30 border border-green-700 rounded-lg">
          <div className="flex items-center gap-2 text-green-300">
            <CheckCircle2 className="w-4 h-4" />
            <span className="font-medium">完成!</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to translate stage names
const getStageLabel = (stage: string): string => {
  const labels: Record<string, string> = {
    init: "初始化",
    fetch_data: "抓取資料",
    train_model: "訓練模型",
    prepare_features: "準備特徵",
    stage_1: "階段 1/4",
    stage_2: "階段 2/4",
    stage_3: "階段 3/4",
    stage_4: "階段 4/4",
    analyze: "分析中",
    completed: "已完成",
  };
  return labels[stage] || stage;
};

// Inline Progress Bar Component (for embedding in other components)
interface InlineProgressProps {
  taskId: string;
  compact?: boolean;
}

export const InlineProgress: React.FC<InlineProgressProps> = ({
  taskId,
  compact = false,
}) => {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);

  useEffect(() => {
    if (!taskId) return;

    const eventSource = new EventSource(`${BACKEND_URL}/progress/${taskId}`);

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressUpdate = JSON.parse(event.data);
        setProgress(data);

        if (data.status === "completed" || data.status === "error") {
          eventSource.close();
        }
      } catch (err) {
        console.error("Error parsing SSE data:", err);
      }
    };

    return () => {
      eventSource.close();
    };
  }, [taskId]);

  if (!progress) return null;

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        {progress.status === "running" && (
          <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
        )}
        {progress.status === "completed" && (
          <CheckCircle2 className="w-4 h-4 text-green-400" />
        )}
        {progress.status === "error" && (
          <XCircle className="w-4 h-4 text-red-400" />
        )}
        <span className="text-sm text-gray-400">
          {progress.percentage.toFixed(0)}% - {progress.message}
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-400">{progress.message}</span>
        <span className="text-purple-300 font-medium">
          {progress.percentage.toFixed(0)}%
        </span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
          style={{ width: `${progress.percentage}%` }}
        />
      </div>
    </div>
  );
};
