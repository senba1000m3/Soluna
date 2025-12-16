// 共鳴配對系統的類型定義

export interface UserProfile {
  name: string;
  avatar: {
    large: string;
  };
}

export interface CommonGenre {
  genre: string;
  score: number;
  similarity: number;
}

export interface CommonAnime {
  id: number;
  title: string;
  coverImage: string;
  user1_score: number;
  user2_score: number;
  score_diff: number;
  average_score: number;
  both_rated: boolean;
}

export interface RadarData {
  labels: string[];
  user1: number[];
  user2: number[];
}

export interface UserStats {
  total_anime: number;
  completed: number;
  avg_score: number;
  episodes_watched: number;
}

export interface Recommendation {
  id: number;
  title: string;
  coverImage: string;
  score: number;
  user_scored: boolean;
  genres: string[];
}

export interface SynergyResponse {
  user1: UserProfile;
  user2: UserProfile;
  compatibility_score: number;
  common_genres: CommonGenre[];
  common_anime: CommonAnime[];
  common_count: number;
  disagreements: CommonAnime[];
  avg_score_difference: number;
  radar_data: RadarData;
  stats: {
    user1: UserStats;
    user2: UserStats;
  };
  recommendations: {
    for_user1: Recommendation[];
    for_user2: Recommendation[];
  };
  message: string;
}

export type SynergyTab = "overview" | "anime" | "recommendations" | "disagreements";
