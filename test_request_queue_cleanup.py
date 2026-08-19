import asyncio
import unittest
from unittest import mock

from RequestQueue import RequestQueue


class RequestQueueCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_timed_out_waiter_closes_late_stream_result(self):
        queue = RequestQueue()
        release_processor = asyncio.Event()
        stream = mock.Mock()

        async def processor(_data, _completion_type):
            await release_processor.wait()
            return stream, None

        request_id = await queue.enqueue_request({}, "chat", processor)
        request = queue.active_requests[request_id]
        process_task = asyncio.create_task(queue._process_request(request))

        with self.assertRaises(asyncio.TimeoutError):
            await queue.wait_for_result(request_id, timeout=0.01)

        release_processor.set()
        await process_task

        stream.close.assert_called_once_with()
        self.assertIsNone(request.result)
        self.assertEqual(queue.processing_count, 0)


if __name__ == "__main__":
    unittest.main()
