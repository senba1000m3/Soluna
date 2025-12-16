"""
SSE 連接診斷腳本
測試進度追蹤是否正常工作
"""

import asyncio
import json
import sys
from datetime import datetime

import httpx


def print_progress(stage: str, message: str = ""):
    """打印進度訊息"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {stage}: {message}")
    sys.stdout.flush()


async def test_sse_connection(
    username: str = "senba1000m3", backend_url: str = "http://localhost:8000"
):
    """
    測試 SSE 連接和進度更新

    Args:
        username: 測試使用者名稱
        backend_url: 後端 URL
    """
    print("\n" + "=" * 70)
    print(f"SSE 連接診斷測試 - 使用者: {username}")
    print(f"後端 URL: {backend_url}")
    print("=" * 70 + "\n")

    # 生成唯一的 task_id
    import random
    import time

    task_id = f"test_{int(time.time())}_{random.randint(1000, 9999)}"

    print_progress("INFO", f"生成 Task ID: {task_id}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        # 階段 1: 啟動分析請求
        print_progress("階段 1", "發送分析請求...")
        try:
            # 不等待回應，直接開始監聽進度
            analyze_task = asyncio.create_task(
                client.post(
                    f"{backend_url}/analyze_drops",
                    json={"username": username, "task_id": task_id},
                )
            )

            # 稍微等待一下確保後端開始處理
            await asyncio.sleep(0.5)

            print_progress("完成", "分析請求已發送，開始監聽進度...")
        except Exception as e:
            print_progress("錯誤", f"發送請求失敗: {e}")
            return

        # 階段 2: 監聽 SSE 進度更新
        print_progress("階段 2", "連接到 SSE 端點...")
        try:
            sse_url = f"{backend_url}/progress/{task_id}"
            print_progress("INFO", f"SSE URL: {sse_url}")

            progress_count = 0
            last_progress = -1
            start_time = time.time()

            async with client.stream("GET", sse_url) as response:
                if response.status_code != 200:
                    print_progress("錯誤", f"SSE 連接失敗: {response.status_code}")
                    return

                print_progress("成功", "SSE 連接已建立")

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]  # 移除 "data: " 前綴

                        try:
                            data = json.loads(data_str)
                            progress = data.get("progress", 0)
                            percentage = data.get("percentage", 0)
                            message = data.get("message", "")
                            stage = data.get("stage", "")
                            status = data.get("status", "")
                            is_heartbeat = data.get("heartbeat", False)

                            # 只顯示進度有變化的更新
                            if progress != last_progress or is_heartbeat:
                                elapsed = time.time() - start_time
                                progress_count += 1

                                heartbeat_indicator = "💓" if is_heartbeat else "📊"
                                print_progress(
                                    f"{heartbeat_indicator} 進度 #{progress_count}",
                                    f"{percentage:.1f}% | {stage} | {status} | {message} | 耗時: {elapsed:.1f}s",
                                )

                                last_progress = progress

                            # 檢查是否完成或錯誤
                            if status == "completed":
                                print_progress(
                                    "成功",
                                    f"任務完成！總共收到 {progress_count} 次更新",
                                )
                                break
                            elif status == "error":
                                print_progress("錯誤", f"任務失敗: {message}")
                                break

                        except json.JSONDecodeError as e:
                            print_progress(
                                "警告", f"無法解析 SSE 數據: {data_str[:100]}"
                            )

                total_time = time.time() - start_time
                print_progress(
                    "統計", f"總耗時: {total_time:.2f}秒，收到 {progress_count} 次更新"
                )

        except httpx.TimeoutException:
            print_progress("錯誤", "SSE 連接超時")
        except Exception as e:
            print_progress("錯誤", f"SSE 監聽失敗: {e}")
            import traceback

            traceback.print_exc()

        # 等待分析任務完成
        try:
            print_progress("階段 3", "等待分析任務回應...")
            result = await asyncio.wait_for(analyze_task, timeout=30.0)

            if result.status_code == 200:
                data = result.json()
                print_progress("成功", f"分析完成！")
                print(f"  ├─ 棄番數量: {data.get('dropped_count', 0)}")
                print(f"  ├─ 正在觀看: {len(data.get('watching_list', []))}")
                print(f"  └─ 預定觀看: {len(data.get('planning_list', []))}")
            else:
                print_progress("錯誤", f"分析失敗: HTTP {result.status_code}")
                try:
                    error_data = result.json()
                    print(f"  錯誤詳情: {error_data.get('detail', 'Unknown')}")
                except:
                    print(f"  回應內容: {result.text[:200]}")

        except asyncio.TimeoutError:
            print_progress("警告", "等待分析結果超時（30秒），但進度已經顯示完成")
        except Exception as e:
            print_progress("錯誤", f"等待分析結果失敗: {e}")

    print("\n" + "=" * 70)
    print("診斷完成")
    print("=" * 70)

    print("\n建議:")
    print("  1. 檢查進度更新是否連續（沒有卡住）")
    print("  2. 檢查是否有收到心跳訊號 (💓)")
    print("  3. 檢查進度百分比是否從 0% 到 100%")
    print("  4. 如果卡在某個進度，檢查後端日誌")


async def main():
    username = sys.argv[1] if len(sys.argv) > 1 else "senba1000m3"
    backend_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    print(f"\n使用者: {username}")
    print(f"後端: {backend_url}")
    print("(可使用參數: python test_sse_connection.py USERNAME BACKEND_URL)")

    try:
        await test_sse_connection(username, backend_url)
    except KeyboardInterrupt:
        print("\n\n測試已中斷")
    except Exception as e:
        print(f"\n\n測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
