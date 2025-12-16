import React from "react";
import { getUserUrl } from "../../utils/synergyHelpers";

interface UserAvatarProps {
  username: string;
  avatarUrl: string;
  size?: "small" | "medium" | "large";
  borderColor: string;
}

/**
 * 用戶頭像鏈接組件
 * 可點擊跳轉到 AniList 用戶頁面
 * 支持不同尺寸和邊框顏色
 */
export const UserAvatar: React.FC<UserAvatarProps> = ({
  username,
  avatarUrl,
  size = "medium",
  borderColor,
}) => {
  const sizeClasses = {
    small: "w-10 h-10",
    medium: "w-12 h-12",
    large: "w-24 h-24",
  };

  return (
    <a
      href={getUserUrl(username)}
      target="_blank"
      rel="noopener noreferrer"
      className={`${sizeClasses[size]} rounded-full border-4 overflow-hidden hover:ring-2 hover:ring-offset-0 transition-all ${borderColor}`}
      title={`查看 ${username} 的 AniList 主頁`}
    >
      <img src={avatarUrl} alt={username} className="w-full h-full object-cover" />
    </a>
  );
};
