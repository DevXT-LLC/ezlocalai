import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedRequest:
    id: str
    data: Dict[str, Any]
    completion_type: str
    timestamp: float
    status: RequestStatus = RequestStatus.QUEUED
    result: Any = None
    error: Optional[str] = None
    future: Optional[asyncio.Future] = None


class RequestQueue:
    def __init__(self, max_concurrent_requests: int = 1, max_queue_size: int = 100):
        """
        Initialize the request queue.

        Args:
            max_concurrent_requests: Maximum number of requests to process simultaneously
            max_queue_size: Maximum number of requests that can be queued
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.processing_count = 0
        self.request_history: Dict[str, QueuedRequest] = {}
        self._task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Metrics
        self.total_processed = 0
        self.total_failed = 0
        self.total_queued = 0

        logging.debug(
            f"[RequestQueue] Initialized with max_concurrent={max_concurrent_requests}, max_queue_size={max_queue_size}"
        )

    async def start(self):
        """Start the queue processor."""
        if self._task is None or self._task.done():
            self._shutdown = False
            self._task = asyncio.create_task(self._process_queue())
            logging.debug("[RequestQueue] Queue processor started")

    async def stop(self):
        """Stop the queue processor gracefully."""
        self._shutdown = True
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            logging.debug("[RequestQueue] Queue processor stopped")

    async def enqueue_request(
        self, data: Dict[str, Any], completion_type: str, processor_func: Callable
    ) -> str:
        """
        Enqueue a new request for processing.

        Args:
            data: Request data to process
            completion_type: Type of completion (chat, completion, etc.)
            processor_func: Function to process the request

        Returns:
            Request ID for tracking

        Raises:
            Exception: If queue is full
        """
        if self.queue.qsize() >= self.max_queue_size:
            try:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=503,
                    detail=f"Request queue is full. Current queue size: {self.queue.qsize()}",
                )
            except ImportError:
                # Fallback for when FastAPI is not available (e.g., in tests)
                raise Exception(
                    f"Request queue is full. Current queue size: {self.queue.qsize()}"
                )

        request_id = str(uuid.uuid4())
        request = QueuedRequest(
            id=request_id,
            data=data,
            completion_type=completion_type,
            timestamp=time.time(),
            future=asyncio.Future(),
        )

        # Store processor function in the request data
        request.processor_func = processor_func

        await self.queue.put(request)
        self.active_requests[request_id] = request
        self.total_queued += 1

        logging.debug(
            f"[RequestQueue] Request {request_id} queued. Queue size: {self.queue.qsize()}"
        )

        return request_id

    async def wait_for_result(
        self, request_id: str, timeout: Optional[float] = None
    ) -> Any:
        """
        Wait for a request to complete and return the result.

        Args:
            request_id: The request ID to wait for
            timeout: Maximum time to wait (None for no timeout)

        Returns:
            The processing result

        Raises:
            Exception: If request not found or failed
            asyncio.TimeoutError: If timeout exceeded
        """
        if request_id not in self.active_requests:
            try:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=404, detail=f"Request {request_id} not found"
                )
            except ImportError:
                raise Exception(f"Request {request_id} not found")

        request = self.active_requests[request_id]

        try:
            if timeout:
                result = await asyncio.wait_for(request.future, timeout=timeout)
            else:
                result = await request.future

            if request.status == RequestStatus.FAILED:
                try:
                    from fastapi import HTTPException

                    raise HTTPException(status_code=500, detail=request.error)
                except ImportError:
                    raise Exception(request.error)

            return result
        except asyncio.TimeoutError:
            logging.warning(
                f"[RequestQueue] Request {request_id} timed out after {timeout}s"
            )
            raise
        finally:
            # Move to history and clean up
            self.request_history[request_id] = self.active_requests.pop(
                request_id, None
            )

    async def _process_queue(self):
        """Main queue processing loop."""
        logging.debug("[RequestQueue] Queue processor loop started")

        while not self._shutdown:
            try:
                # Wait for requests while respecting concurrency limits
                if self.processing_count >= self.max_concurrent_requests:
                    await asyncio.sleep(0.1)
                    continue

                # Get next request with timeout to allow shutdown checks
                try:
                    request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process request in background
                asyncio.create_task(self._process_request(request))

                # Small delay to allow processing_count to update
                await asyncio.sleep(0.01)

            except Exception as e:
                logging.error(f"[RequestQueue] Error in queue processor: {e}")
                await asyncio.sleep(1.0)

    async def _process_request(self, request: QueuedRequest):
        """Process a single request."""
        self.processing_count += 1
        request.status = RequestStatus.PROCESSING

        start_time = time.time()
        logging.debug(
            f"[RequestQueue] Processing request {request.id} (type: {request.completion_type})"
        )

        try:
            # Call the processor function
            result = await request.processor_func(request.data, request.completion_type)

            request.status = RequestStatus.COMPLETED
            request.result = result
            request.future.set_result(result)
            self.total_processed += 1

            processing_time = time.time() - start_time
            logging.debug(
                f"[RequestQueue] Request {request.id} completed in {processing_time:.2f}s"
            )

        except Exception as e:
            request.status = RequestStatus.FAILED
            request.error = str(e)
            request.future.set_exception(e)
            self.total_failed += 1

            processing_time = time.time() - start_time
            logging.error(
                f"[RequestQueue] Request {request.id} failed after {processing_time:.2f}s: {e}"
            )

        finally:
            self.processing_count -= 1

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and metrics."""
        return {
            "queue_size": self.queue.qsize(),
            "processing_count": self.processing_count,
            "max_concurrent": self.max_concurrent_requests,
            "max_queue_size": self.max_queue_size,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_queued": self.total_queued,
            "active_requests": len(self.active_requests),
            "is_running": self._task is not None and not self._task.done(),
        }

    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request."""
        request = self.active_requests.get(request_id) or self.request_history.get(
            request_id
        )

        if not request:
            return None

        return {
            "id": request.id,
            "status": request.status.value,
            "completion_type": request.completion_type,
            "timestamp": request.timestamp,
            "error": request.error,
        }
