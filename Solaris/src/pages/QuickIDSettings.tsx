import React, { useState } from "react";
import { useGlobalUser } from "../contexts/AuthContext";
import {
  Plus,
  Trash2,
  Star,
  Loader2,
  ExternalLink,
  User,
  AlertCircle,
  Edit2,
  Save,
  X,
} from "lucide-react";

export const QuickIDSettings = () => {
  const {
    mainUser,
    quickIds,
    addQuickId,
    removeQuickId,
    updateQuickIdNickname,
  } = useGlobalUser();
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editNickname, setEditNickname] = useState("");

  // 新增表單狀態
  const [newAnilistUsername, setNewAnilistUsername] = useState("");
  const [newNickname, setNewNickname] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    try {
      await addQuickId(newAnilistUsername, newNickname || undefined);
      // 重置表單
      setNewAnilistUsername("");
      setNewNickname("");
      setShowAddForm(false);
      setSuccess("成功新增常用 ID！");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err: any) {
      setError(err.message || "新增失敗，請檢查使用者名稱是否正確");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("確定要刪除此常用 ID 嗎？")) return;

    setError("");
    setLoading(true);

    try {
      await removeQuickId(id);
      setSuccess("成功刪除常用 ID！");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err: any) {
      setError(err.message || "刪除失敗");
    } finally {
      setLoading(false);
    }
  };

  const startEdit = (qid: any) => {
    setEditingId(qid.id);
    setEditNickname(qid.nickname || "");
  };

  const handleSaveEdit = async (id: number) => {
    setError("");
    setLoading(true);

    try {
      await updateQuickIdNickname(id, editNickname);
      setEditingId(null);
      setSuccess("成功更新暱稱！");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err: any) {
      setError(err.message || "更新失敗");
    } finally {
      setLoading(false);
    }
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditNickname("");
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
          快速 ID 管理
        </h1>
        <p className="text-gray-400">
          管理你的主 ID 和常用 ID 列表，讓查詢更加便利
        </p>
      </div>

      {/* 主 ID 顯示 */}
      {mainUser && (
        <div className="mb-6 bg-gradient-to-r from-purple-900/40 to-pink-900/40 border border-purple-700/50 p-6 rounded-xl">
          <div className="flex items-center gap-2 mb-3">
            <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
            <h2 className="text-lg font-bold text-white">
              主 ID（全局使用者）
            </h2>
          </div>
          <div className="flex items-center gap-4">
            <img
              src={mainUser.avatar}
              alt={mainUser.anilistUsername}
              className="w-16 h-16 rounded-full object-cover border-2 border-purple-400"
            />
            <div className="flex-1">
              <div className="font-bold text-xl text-white">
                {mainUser.anilistUsername}
              </div>
              <div className="text-sm text-gray-400">
                ID: {mainUser.anilistId}
              </div>
              <a
                href={`https://anilist.co/user/${mainUser.anilistUsername}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-purple-400 hover:text-purple-300 transition-colors inline-flex items-center gap-1 mt-1"
              >
                前往 AniList 個人頁面
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            💡 主 ID 是你的全局使用者，會出現在所有頁面的快速選擇列表中
          </p>
        </div>
      )}

      {/* 未設定主 ID 的提示 */}
      {!mainUser && (
        <div className="mb-6 bg-yellow-900/20 border border-yellow-700 p-6 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <h3 className="font-bold text-yellow-400">尚未設定主 ID</h3>
          </div>
          <p className="text-gray-300 mb-3">
            請先點擊右上角的「設定使用者」按鈕來設定你的主 ID
          </p>
          <p className="text-sm text-gray-400">
            主 ID 設定後，你就可以在這裡新增其他常用的 AniList 使用者 ID 了
          </p>
        </div>
      )}

      {/* 成功/錯誤訊息 */}
      {success && (
        <div className="mb-6 text-green-400 bg-green-900/20 p-4 rounded-lg border border-green-800 flex items-center gap-2">
          <Star className="w-5 h-5" />
          {success}
        </div>
      )}

      {error && (
        <div className="mb-6 text-red-400 bg-red-900/20 p-4 rounded-lg border border-red-800 flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {/* 新增按鈕（只有設定主 ID 後才顯示） */}
      {mainUser && !showAddForm && (
        <button
          onClick={() => setShowAddForm(true)}
          className="w-full mb-6 px-6 py-4 bg-purple-600 hover:bg-purple-700 rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
        >
          <Plus className="w-5 h-5" />
          新增常用 ID
        </button>
      )}

      {/* 新增表單 */}
      {showAddForm && (
        <div className="mb-6 bg-gray-800 p-6 rounded-xl border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-purple-300">新增常用 ID</h2>
            <button
              onClick={() => {
                setShowAddForm(false);
                setError("");
              }}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>

          <form onSubmit={handleAdd} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                AniList 使用者名稱 *
              </label>
              <input
                type="text"
                value={newAnilistUsername}
                onChange={(e) => setNewAnilistUsername(e.target.value)}
                required
                className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
                placeholder="例如: thet"
              />
              <p className="text-xs text-gray-500 mt-1">
                系統會自動從 AniList 抓取 ID 和頭像
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                自訂暱稱 (選填)
              </label>
              <input
                type="text"
                value={newNickname}
                onChange={(e) => setNewNickname(e.target.value)}
                className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
                placeholder="例如: 朋友的帳號"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  新增中...
                </>
              ) : (
                <>
                  <Plus className="w-5 h-5" />
                  新增
                </>
              )}
            </button>
          </form>
        </div>
      )}

      {/* 常用 ID 列表 */}
      {mainUser && (
        <div>
          <h2 className="text-xl font-bold mb-4 text-white flex items-center gap-2">
            <User className="w-5 h-5" />
            常用 ID 列表
            <span className="text-sm text-gray-400 font-normal">
              ({quickIds.length} 個)
            </span>
          </h2>

          <div className="space-y-4">
            {quickIds.length === 0 ? (
              <div className="text-center py-12 text-gray-500 bg-gray-800/50 rounded-xl border border-gray-700">
                <User className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>還沒有常用 ID</p>
                <p className="text-sm mt-1">
                  點擊上方按鈕新增其他常用的 AniList ID
                </p>
              </div>
            ) : (
              quickIds.map((qid) => (
                <div
                  key={qid.id}
                  className="bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-gray-600 transition-all"
                >
                  {editingId === qid.id ? (
                    // 編輯模式
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium mb-2 text-gray-300">
                          自訂暱稱
                        </label>
                        <input
                          type="text"
                          value={editNickname}
                          onChange={(e) => setEditNickname(e.target.value)}
                          className="w-full px-4 py-2 bg-gray-700 rounded-lg border border-gray-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500 outline-none transition-all"
                          placeholder="輸入暱稱"
                        />
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={() => handleSaveEdit(qid.id)}
                          disabled={loading}
                          className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          <Save className="w-4 h-4" />
                          儲存
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                        >
                          <X className="w-4 h-4" />
                          取消
                        </button>
                      </div>
                    </div>
                  ) : (
                    // 顯示模式
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-4 flex-1">
                        <img
                          src={qid.avatar}
                          alt={qid.anilistUsername}
                          className="w-16 h-16 rounded-full object-cover"
                        />
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="text-lg font-bold text-white">
                              {qid.nickname || qid.anilistUsername}
                            </h3>
                          </div>
                          {qid.nickname && (
                            <p className="text-sm text-gray-400 mb-1">
                              使用者名稱: @{qid.anilistUsername}
                            </p>
                          )}
                          <div className="flex items-center gap-3">
                            <p className="text-sm text-gray-400">
                              ID: {qid.anilistId}
                            </p>
                            <a
                              href={`https://anilist.co/user/${qid.anilistUsername}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-purple-400 hover:text-purple-300 transition-colors"
                            >
                              <ExternalLink className="w-4 h-4" />
                            </a>
                          </div>
                          <p className="text-xs text-gray-500 mt-2">
                            新增時間:{" "}
                            {new Date(qid.createdAt).toLocaleDateString(
                              "zh-TW",
                            )}
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={() => startEdit(qid)}
                          className="p-2 text-blue-400 hover:bg-gray-700 rounded-lg transition-colors"
                          title="編輯暱稱"
                        >
                          <Edit2 className="w-5 h-5" />
                        </button>
                        <button
                          onClick={() => handleDelete(qid.id)}
                          disabled={loading}
                          className="p-2 text-red-400 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
                          title="刪除"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* 使用說明 */}
      <div className="mt-8 bg-gray-800/50 p-6 rounded-xl border border-gray-700">
        <h3 className="font-bold text-lg mb-3 text-purple-300">使用說明</h3>
        <ul className="space-y-2 text-sm text-gray-400">
          <li className="flex items-start gap-2">
            <span className="text-purple-400 mt-0.5">•</span>
            <span>
              <strong className="text-white">主 ID</strong> -
              你的全局使用者，會出現在所有頁面的快速選擇列表中
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-400 mt-0.5">•</span>
            <span>
              <strong className="text-white">常用 ID 列表</strong> -
              可以新增其他常用的 AniList ID（朋友、小號等）
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-400 mt-0.5">•</span>
            <span>
              <strong className="text-white">快速選擇</strong> -
              在各功能頁面的輸入框旁邊點擊快速選擇按鈕，即可選擇主 ID 或常用 ID
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-400 mt-0.5">•</span>
            <span>
              <strong className="text-white">登出</strong> -
              點擊右上角的登出按鈕會清除主 ID 及所有常用 ID
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-400 mt-0.5">•</span>
            <span>
              <strong className="text-white">資料持久化</strong> -
              所有資料儲存在資料庫中，下次登入相同 ID 會自動恢復
            </span>
          </li>
        </ul>
      </div>

      {/* 額外資訊 */}
      <div className="mt-6 p-4 bg-blue-900/20 border border-blue-700 rounded-lg">
        <h4 className="font-bold text-blue-300 mb-2 flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          注意事項
        </h4>
        <ul className="text-sm text-blue-200 space-y-1">
          <li>• 此功能查詢的是 AniList 的公開資料，無需密碼</li>
          <li>• 資料儲存在 soluna.db 資料庫中，安全可靠</li>
          <li>• 主 ID 無法直接刪除，請使用右上角的登出功能</li>
          <li>• 常用 ID 不會覆蓋主 ID，兩者互相獨立</li>
        </ul>
      </div>
    </div>
  );
};
