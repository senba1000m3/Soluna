import React from "react";
import { AlertCircle } from "lucide-react";
import { CommonAnime, UserProfile } from "../../types/synergy";
import { getAnimeUrl, formatScore } from "../../utils/synergyHelpers";
import { UserAvatar } from "./UserAvatar";

interface DisagreementsModalProps {
  isOpen: boolean;
  onClose: () => void;
  disagreements: CommonAnime[];
  user1: UserProfile;
  user2: UserProfile;
}

/**
 * 品味分歧彈窗組件
 * 顯示所有評分差異較大的共同作品
 * 包含滾動列表和關閉功能
 */
export const DisagreementsModal: React.FC<DisagreementsModalProps> = ({
  isOpen,
  onClose,
  disagreements,
  user1,
  user2,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-2xl p-8 border border-orange-500/30 shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 標題欄 */}
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-700">
          <h3 className="text-2xl font-bold flex items-center gap-2">
            <AlertCircle className="w-6 h-6 text-orange-400" />
            所有品味分歧作品 ({disagreements.length})
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white hover:bg-gray-700 transition-all text-2xl font-bold w-10 h-10 flex items-center justify-center rounded-lg"
            title="關閉"
          >
            ×
          </button>
        </div>

        {/* 滾動內容區 */}
        <div className="space-y-4 overflow-y-auto pr-2 flex-1">
          {disagreements.length > 0 ? (
            disagreements.map((anime, idx) => (
              <DisagreementItem
                key={anime.id}
                anime={anime}
                index={idx}
                user1={user1}
                user2={user2}
              />
            ))
          ) : (
            <EmptyState />
          )}
        </div>

        {/* 底部提示 */}
        <div className="mt-4 pt-4 border-t border-gray-700 text-center text-sm text-gray-400">
          使用滑鼠滾輪或觸控滑動瀏覽 • 點擊作品查看詳情
        </div>
      </div>
    </div>
  );
};

/**
 * 單個分歧項目組件
 */
const DisagreementItem: React.FC<{
  anime: CommonAnime;
  index: number;
  user1: UserProfile;
  user2: UserProfile;
}> = ({ anime, index, user1, user2 }) => {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 flex flex-col md:flex-row gap-4 items-start md:items-center hover:bg-gray-700/70 transition-colors">
      {/* 排名 */}
      <div className="text-2xl font-bold text-gray-500 w-8">#{index + 1}</div>

      {/* 動畫封面 */}
      <a
        href={getAnimeUrl(anime.id)}
        target="_blank"
        rel="noopener noreferrer"
      >
        <img
          src={anime.coverImage}
          alt={anime.title}
          className="w-16 h-24 object-cover rounded hover:ring-2 hover:ring-orange-400 transition-all"
        />
      </a>

      {/* 動畫信息和評分對比 */}
      <div className="flex-1">
        <a
          href={getAnimeUrl(anime.id)}
          target="_blank"
          rel="noopener noreferrer"
          className="font-bold text-lg mb-2 hover:text-blue-400 transition-colors block"
        >
          {anime.title}
        </a>

        <div className="flex items-center gap-6">
          {/* 用戶1評分 */}
          <UserScore
            username={user1.name}
            avatarUrl={user1.avatar.large}
            score={anime.user1_score}
            borderColor="border-blue-500 hover:ring-blue-400"
            scoreColor="text-blue-400"
          />

          {/* 用戶2評分 */}
          <UserScore
            username={user2.name}
            avatarUrl={user2.avatar.large}
            score={anime.user2_score}
            borderColor="border-teal-500 hover:ring-teal-400"
            scoreColor="text-teal-400"
          />

          {/* 評分差異 */}
          <div className="ml-auto">
            <div className="text-xs text-gray-400">評分差異</div>
            <div className="text-2xl font-bold text-orange-400">
              {anime.score_diff.toFixed(2)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * 用戶評分顯示組件
 */
const UserScore: React.FC<{
  username: string;
  avatarUrl: string;
  score: number;
  borderColor: string;
  scoreColor: string;
}> = ({ username, avatarUrl, score, borderColor, scoreColor }) => {
  return (
    <div className="flex items-center gap-2">
      <UserAvatar
        username={username}
        avatarUrl={avatarUrl}
        size="small"
        borderColor={borderColor}
      />
      <div>
        <div className="text-xs text-gray-400">{username}</div>
        <div className={`text-xl font-bold ${scoreColor}`}>
          {formatScore(score, 1)}
        </div>
      </div>
    </div>
  );
};

/**
 * 空狀態組件
 */
const EmptyState: React.FC = () => {
  return (
    <div className="text-center py-12 text-gray-400">
      <AlertCircle className="w-16 h-16 mx-auto mb-4 opacity-50" />
      <p className="text-lg">沒有找到評分差異的作品</p>
      <p className="text-sm mt-2">可能雙方都沒有對共同作品評分</p>
    </div>
  );
};
