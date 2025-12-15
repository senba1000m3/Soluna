"""
Progress Tracker for Real-time Progress Updates
Provides a thread-safe way to track and broadcast progress updates
"""

import logging
import threading
import time
from queue import Queue
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Thread-safe progress tracker for broadcasting updates to clients
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.progress_queue: Queue = Queue()
        self.current_progress = 0
        self.total_progress = 100
        self.status = "pending"
        self.message = ""
        self.stage = ""
        self.lock = threading.Lock()

    def update(
        self,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
    ):
        """
        Update progress and broadcast to listeners

        Args:
            progress: Current progress value (0-100)
            message: Progress message
            stage: Current stage name
            status: Status (pending, running, completed, error)
        """
        with self.lock:
            if progress is not None:
                self.current_progress = min(progress, self.total_progress)
            if message is not None:
                self.message = message
            if stage is not None:
                self.stage = stage
            if status is not None:
                self.status = status

            update_data = {
                "task_id": self.task_id,
                "progress": self.current_progress,
                "total": self.total_progress,
                "percentage": round(
                    (self.current_progress / self.total_progress) * 100, 1
                ),
                "message": self.message,
                "stage": self.stage,
                "status": self.status,
                "timestamp": time.time(),
            }

            self.progress_queue.put(update_data)
            return update_data

    def increment(self, amount: int = 1, message: Optional[str] = None):
        """
        Increment progress by amount

        Args:
            amount: Amount to increment
            message: Optional message
        """
        with self.lock:
            self.current_progress = min(
                self.current_progress + amount, self.total_progress
            )
            if message is not None:
                self.message = message

            update_data = {
                "task_id": self.task_id,
                "progress": self.current_progress,
                "total": self.total_progress,
                "percentage": round(
                    (self.current_progress / self.total_progress) * 100, 1
                ),
                "message": self.message,
                "stage": self.stage,
                "status": self.status,
                "timestamp": time.time(),
            }

            self.progress_queue.put(update_data)
            return update_data

    def complete(self, message: str = "完成"):
        """Mark task as completed"""
        return self.update(
            progress=self.total_progress, message=message, status="completed"
        )

    def error(self, message: str = "發生錯誤"):
        """Mark task as error"""
        return self.update(message=message, status="error")

    def get_updates(self):
        """
        Generator that yields progress updates

        Yields:
            Progress update dictionaries
        """
        while True:
            try:
                update = self.progress_queue.get(timeout=30)
                yield update

                # Stop if completed or error
                if update["status"] in ["completed", "error"]:
                    break
            except:
                # Timeout - send heartbeat
                with self.lock:
                    yield {
                        "task_id": self.task_id,
                        "progress": self.current_progress,
                        "total": self.total_progress,
                        "percentage": round(
                            (self.current_progress / self.total_progress) * 100, 1
                        ),
                        "message": self.message,
                        "stage": self.stage,
                        "status": self.status,
                        "timestamp": time.time(),
                        "heartbeat": True,
                    }


class ProgressManager:
    """
    Global progress manager to track multiple tasks
    """

    def __init__(self):
        self.tasks: Dict[str, ProgressTracker] = {}
        self.lock = threading.Lock()

    def create_task(self, task_id: str) -> ProgressTracker:
        """Create a new progress tracker for a task"""
        with self.lock:
            tracker = ProgressTracker(task_id)
            self.tasks[task_id] = tracker
            logger.info(f"Created progress tracker for task: {task_id}")
            return tracker

    def get_task(self, task_id: str) -> Optional[ProgressTracker]:
        """Get progress tracker for a task"""
        with self.lock:
            return self.tasks.get(task_id)

    def remove_task(self, task_id: str):
        """Remove a completed task"""
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"Removed progress tracker for task: {task_id}")


# Global progress manager instance
progress_manager = ProgressManager()
