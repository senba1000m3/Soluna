"""
快速測試腳本 - 驗證 SSE 和 API 是否正常工作
"""

import asyncio
import json
import sys
import time

import httpx


async def test_api(username: str = "TheT", backend_url: str = "http://localhost:8000"):
    """測試 analyze_drops API 和 SSE 連接"""
    print("\n" + "=" * 70)
    print(f"測試棄番預測 API - 使用者: {username}")
    print(f"後端 URL: {backend_url}")
    print("=" * 70 + "\n")

    # 生成 task_id
    task_id = f"drop_{int(time.time())}_{int(time.time() * 1000) % 1000}"
    print(f"Task ID: {task_id}\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 同時啟動兩個任務
        print("📡 同時啟動 SSE 監聽和 API 請求...\n")

        # 任務 1: 監聽 SSE 進度
        async def listen_progress():
            print("🎧 [SSE] 開始監聽進度...")
            sse_url = f"{backend_url}/progress/{task_id}"
            updates_received = 0

            try:
                async with client.stream("GET", sse_url) as response:
                    if response.status_code != 200:
                        print(f"❌ [SSE] 連接失敗: {response.status_code}")
                        return

                    print(f"✅ [SSE] 連接成功！\n")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                            updates_received += 1

                            progress = data.get("progress", 0)
                            percentage = data.get("percentage", 0)
                            message = data.get("message", "")
                            status = data.get("status", "")
                            stage = data.get("stage", "")
                            is_heartbeat = data.get("heartbeat", False)

                            icon = "💓" if is_heartbeat else "📊"
                            print(
                                f"{icon} [SSE #{updates_received:02d}] {percentage:5.1f}% | {status:10s} | {stage:15s} | {message}"
                            )

                            if status in ["completed", "error"]:
                                print(
                                    f"\n✅ [SSE] 任務 {status}，共收到 {updates_received} 次更新"
                                )
                                break

                        except json.JSONDecodeError:
                            print(f"⚠️  [SSE] 無法解析: {line[:80]}")

            except Exception as e:
                print(f"❌ [SSE] 錯誤: {e}")
                import traceback

                traceback.print_exc()

        # 任務 2: 發送 analyze_drops 請求
        async def call_api():
            # 稍微延遲，讓 SSE 先連接
            await asyncio.sleep(0.1)

            print("📤 [API] 發送分析請求...\n")

            try:
                response = await client.post(
                    f"{backend_url}/analyze_drops",
                    json={"username": username, "task_id": task_id},
                )

                if response.status_code == 200:
                    data = response.json()
                    print("\n" + "=" * 70)
                    print("✅ [API] 分析完成！")
                    print("=" * 70)
                    print(f"棄番數量: {data.get('dropped_count', 0)}")
                    print(f"正在觀看: {len(data.get('watching_list', []))}")
                    print(f"預定觀看: {len(data.get('planning_list', []))}")

                    # 顯示前 3 個高風險動畫
                    watching = data.get("watching_list", [])
                    if watching:
                        print("\n高風險動畫 (前3):")
                        for i, anime in enumerate(watching[:3], 1):
                            prob = anime.get("drop_probability", 0)
                            if prob and prob > 0:
                                print(
                                    f"  {i}. {anime.get('title', 'Unknown')} - {prob:.1%}"
                                )
                else:
                    error = response.json().get("detail", "Unknown error")
                    print(f"\n❌ [API] 失敗: {response.status_code} - {error}")

            except Exception as e:
                print(f"\n❌ [API] 錯誤: {e}")
                import traceback

                traceback.print_exc()

        # 同時執行
        await asyncio.gather(
            listen_progress(),
            call_api(),
        )

    print("\n" + "=" * 70)
    print("測試完成")
    print("=" * 70)


async def test_simple(
    username: str = "TheT", backend_url: str = "http://localhost:8000"
):
    """簡單測試 - 只調用 API 不監聽 SSE"""
    print("\n簡單測試 - 只調用 API\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        task_id = f"simple_{int(time.time())}"

        print(f"發送請求... (使用者: {username}, task_id: {task_id})")

        try:
            response = await client.post(
                f"{backend_url}/analyze_drops",
                json={"username": username, "task_id": task_id},
            )

            print(f"狀態碼: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(
                    f"✅ 成功！棄番: {data.get('dropped_count')}, 觀看: {len(data.get('watching_list', []))}"
                )
            else:
                print(f"❌ 失敗: {response.text[:200]}")

        except Exception as e:
            print(f"❌ 錯誤: {e}")


async def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python quick_test.py <username> [backend_url] [mode]")
        print("\n模式:")
        print("  full   - 完整測試 (SSE + API, 預設)")
        print("  simple - 簡單測試 (只有 API)")
        print("\n範例:")
        print("  python quick_test.py TheT")
        print("  python quick_test.py senba1000m3 http://localhost:8000")
        print("  python quick_test.py TheT http://localhost:8000 simple")
        return

    username = sys.argv[1]
    backend_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    mode = sys.argv[3] if len(sys.argv) > 3 else "full"

    if mode == "simple":
        await test_simple(username, backend_url)
    else:
        await test_api(username, backend_url)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n測試已中斷")
    except Exception as e:
        print(f"\n\n測試失敗: {e}")
        import traceback

        traceback.print_exc()
