// 共鳴配對系統的工具函數

/**
 * 根據共鳴分數獲取對應的顏色類名
 * @param score - 共鳴分數 (0-100)
 * @returns Tailwind CSS 顏色類名
 */
export const getScoreColor = (score: number): string => {
  if (score >= 80) return "text-green-400";
  if (score >= 60) return "text-yellow-400";
  return "text-red-400";
};

/**
 * 根據共鳴分數獲取對應的評語
 * @param score - 共鳴分數 (0-100)
 * @returns 評語文字
 */
export const getScoreMessage = (score: number): string => {
  if (score >= 90) return "靈魂伴侶！你們的品味驚人地相似！";
  if (score >= 80) return "非常合拍！有很多共同話題。";
  if (score >= 60) return "還不錯，有些共同喜好。";
  if (score >= 40) return "品味有些差異，但可以互相推坑。";
  return "水火不容？或許是互補的關係！";
};

/**
 * 檢查分數是否有效（大於 0）
 * @param score - 分數值
 * @returns 是否有效
 */
export const isValidScore = (score: number): boolean => {
  return score > 0;
};

/**
 * 格式化分數顯示（無效分數顯示為 "-"）
 * @param score - 分數值
 * @param decimals - 小數位數，默認為 1
 * @returns 格式化後的字符串
 */
export const formatScore = (score: number, decimals: number = 1): string => {
  return isValidScore(score) ? score.toFixed(decimals) : "-";
};

/**
 * 檢查是否應該顯示評分差異
 * @param score1 - 用戶1的評分
 * @param score2 - 用戶2的評分
 * @returns 是否應該顯示差異
 */
export const shouldShowScoreDiff = (score1: number, score2: number): boolean => {
  return isValidScore(score1) && isValidScore(score2);
};

/**
 * 生成 AniList 動畫頁面 URL
 * @param animeId - 動畫 ID
 * @returns 完整 URL
 */
export const getAnimeUrl = (animeId: number): string => {
  return `https://anilist.co/anime/${animeId}`;
};

/**
 * 生成 AniList 用戶頁面 URL
 * @param username - 用戶名
 * @returns 完整 URL
 */
export const getUserUrl = (username: string): string => {
  return `https://anilist.co/user/${username}`;
};

/**
 * 檢查推薦列表是否為空
 * @param recommendations - 推薦列表
 * @returns 是否為空
 */
export const hasRecommendations = (recommendations: any[]): boolean => {
  return recommendations && recommendations.length > 0;
};

/**
 * 檢查是否有品味分歧數據
 * @param disagreements - 分歧列表
 * @returns 是否有數據
 */
export const hasDisagreements = (disagreements: any[]): boolean => {
  return disagreements && disagreements.length > 0;
};

/**
 * 檢查是否應該顯示"查看更多"按鈕
 * @param items - 項目列表
 * @param threshold - 閾值，默認為 5
 * @returns 是否應該顯示
 */
export const shouldShowViewMore = (items: any[], threshold: number = 5): boolean => {
  return items && items.length > threshold;
};
