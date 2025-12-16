import React, { createContext, useContext, useState, useEffect } from "react";
import axios from "axios";
import { BACKEND_URL } from "../config/env";

interface AniListUser {
  id: number;
  name: string;
  avatar: {
    large: string;
    medium: string;
  };
  bannerImage?: string;
}

interface MainUser {
  id: number;
  anilistUsername: string;
  anilistId: number;
  avatar: string;
  createdAt: string;
}

interface QuickID {
  id: number;
  anilistUsername: string;
  anilistId: number;
  avatar: string;
  nickname?: string;
  createdAt: string;
}

interface GlobalUserContextType {
  mainUser: MainUser | null;
  quickIds: QuickID[];
  loading: boolean;
  loginUser: (username: string) => Promise<void>;
  logoutUser: () => Promise<void>;
  addQuickId: (username: string, nickname?: string) => Promise<void>;
  removeQuickId: (id: number) => Promise<void>;
  updateQuickIdNickname: (id: number, nickname: string) => Promise<void>;
}

const GlobalUserContext = createContext<GlobalUserContextType | undefined>(
  undefined,
);

export const useGlobalUser = () => {
  const context = useContext(GlobalUserContext);
  if (!context) {
    throw new Error("useGlobalUser must be used within GlobalUserProvider");
  }
  return context;
};

const STORAGE_KEY = "soluna_main_user";

export const GlobalUserProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [mainUser, setMainUser] = useState<MainUser | null>(null);
  const [quickIds, setQuickIds] = useState<QuickID[]>([]);
  const [loading, setLoading] = useState(false);

  // 從 localStorage 載入主 ID
  useEffect(() => {
    const storedUser = localStorage.getItem(STORAGE_KEY);
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setMainUser(user);
        // 從後端載入常用 ID (異步執行)
        loadQuickIds(user.anilistId).catch((err) => {
          console.error("Failed to load quick IDs on startup:", err);
        });
      } catch (e) {
        console.error("Failed to parse stored user:", e);
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  // 從 AniList API 獲取使用者資訊
  const fetchAniListUser = async (username: string): Promise<AniListUser> => {
    const query = `
      query ($name: String) {
        User(name: $name) {
          id
          name
          avatar {
            large
            medium
          }
          bannerImage
        }
      }
    `;

    const response = await axios.post(
      "https://graphql.anilist.co",
      {
        query,
        variables: { name: username },
      },
      {
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
      },
    );

    if (response.data.errors) {
      throw new Error(response.data.errors[0].message);
    }

    return response.data.data.User;
  };

  // 從後端載入常用 ID 列表
  const loadQuickIds = async (anilistId: number) => {
    try {
      console.log(`Loading quick IDs for user ${anilistId}...`);
      const response = await axios.get(
        `${BACKEND_URL}/global-user/${anilistId}/quick-ids`,
      );
      console.log(`Loaded ${response.data.length} quick IDs`);
      setQuickIds(response.data);
    } catch (error) {
      console.error("Failed to load quick IDs:", error);
      // 即使失敗也設置為空數組，避免 undefined
      setQuickIds([]);
    }
  };

  // 登入使用者（設定主 ID）
  const loginUser = async (username: string) => {
    if (!username.trim()) {
      throw new Error("請輸入使用者名稱");
    }

    setLoading(true);
    try {
      // 從 AniList 獲取使用者資訊
      const anilistUser = await fetchAniListUser(username);

      // 登入到後端（如果已存在會返回現有資料）
      const response = await axios.post(`${BACKEND_URL}/global-user/login`, {
        anilist_username: anilistUser.name,
        anilist_id: anilistUser.id,
        avatar: anilistUser.avatar.large,
      });

      const user = response.data.user;
      setMainUser(user);
      setQuickIds(response.data.quickIds);

      // 儲存到 localStorage
      localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    } catch (error: any) {
      console.error("Failed to login user:", error);
      throw new Error(error.response?.data?.detail || "找不到此使用者");
    } finally {
      setLoading(false);
    }
  };

  // 登出使用者（只清除前端狀態，保留資料庫資料）
  const logoutUser = async () => {
    if (!mainUser) return;

    // 只清除前端的 localStorage 和狀態
    // 資料保留在資料庫中，下次登入相同 ID 時會自動恢復
    setMainUser(null);
    setQuickIds([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  // 新增常用 ID
  const addQuickId = async (username: string, nickname?: string) => {
    if (!mainUser) {
      throw new Error("請先設定主 ID");
    }

    setLoading(true);
    try {
      // 從 AniList 獲取使用者資訊
      const anilistUser = await fetchAniListUser(username);

      // 檢查是否為主 ID 本身
      if (anilistUser.id === mainUser.anilistId) {
        throw new Error("不能將主 ID 加入常用列表");
      }

      // 檢查是否已存在
      const exists = quickIds.some((qid) => qid.anilistId === anilistUser.id);
      if (exists) {
        throw new Error("此使用者已在常用列表中");
      }

      // 新增到後端
      const response = await axios.post(`${BACKEND_URL}/quick-ids`, {
        owner_anilist_id: mainUser.anilistId,
        anilist_username: anilistUser.name,
        anilist_id: anilistUser.id,
        avatar: anilistUser.avatar.large,
        nickname,
      });

      // 更新前端狀態
      setQuickIds([...quickIds, response.data]);
    } catch (error: any) {
      console.error("Failed to add quick ID:", error);
      throw new Error(
        error.response?.data?.detail || error.message || "新增失敗",
      );
    } finally {
      setLoading(false);
    }
  };

  // 刪除常用 ID
  const removeQuickId = async (id: number) => {
    setLoading(true);
    try {
      await axios.delete(`${BACKEND_URL}/quick-ids/${id}`);
      setQuickIds(quickIds.filter((qid) => qid.id !== id));
    } catch (error) {
      console.error("Failed to remove quick ID:", error);
      throw new Error("刪除失敗");
    } finally {
      setLoading(false);
    }
  };

  // 更新常用 ID 的暱稱
  const updateQuickIdNickname = async (id: number, nickname: string) => {
    setLoading(true);
    try {
      const response = await axios.patch(
        `${BACKEND_URL}/quick-ids/${id}`,
        null,
        {
          params: { nickname },
        },
      );
      setQuickIds(quickIds.map((qid) => (qid.id === id ? response.data : qid)));
    } catch (error) {
      console.error("Failed to update quick ID:", error);
      throw new Error("更新失敗");
    } finally {
      setLoading(false);
    }
  };

  return (
    <GlobalUserContext.Provider
      value={{
        mainUser,
        quickIds,
        loading,
        loginUser,
        logoutUser,
        addQuickId,
        removeQuickId,
        updateQuickIdNickname,
      }}
    >
      {children}
    </GlobalUserContext.Provider>
  );
};

// 為了向後兼容，保留 useAuth 別名
export const useAuth = useGlobalUser;
export const AuthProvider = GlobalUserProvider;
