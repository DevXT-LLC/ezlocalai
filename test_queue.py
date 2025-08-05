#!/usr/bin/env python3
"""
Test script for the Request Queue system.
This script tests the queue functionality without requiring a full ezlocalai setup.
"""

import asyncio
import time
import logging
from RequestQueue import RequestQueue, RequestStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def mock_processor(data, completion_type):
    """Mock processor function that simulates AI processing."""
    processing_time = data.get('processing_time', 1)
    await asyncio.sleep(processing_time)
    
    if data.get('should_fail', False):
        raise Exception("Mock processing failure")
    
    return {
        "choices": [{"message": {"content": f"Response for {completion_type}: {data.get('prompt', 'test')}"}}]
    }, None

async def test_queue_basic():
    """Test basic queue functionality."""
    print("\n=== Testing Basic Queue Functionality ===")
    
    queue = RequestQueue(max_concurrent_requests=2, max_queue_size=5)
    await queue.start()
    
    try:
        # Test single request
        request_id = await queue.enqueue_request(
            data={"prompt": "Hello", "processing_time": 0.5},
            completion_type="chat",
            processor_func=mock_processor
        )
        
        result = await queue.wait_for_result(request_id, timeout=5.0)
        print(f"✓ Single request completed: {result}")
        
        # Test queue status
        status = queue.get_queue_status()
        print(f"✓ Queue status: {status}")
        
    finally:
        await queue.stop()

async def test_queue_concurrent():
    """Test concurrent request processing."""
    print("\n=== Testing Concurrent Processing ===")
    
    queue = RequestQueue(max_concurrent_requests=2, max_queue_size=10)
    await queue.start()
    
    try:
        # Submit multiple requests
        request_ids = []
        start_time = time.time()
        
        for i in range(4):
            request_id = await queue.enqueue_request(
                data={"prompt": f"Request {i}", "processing_time": 1.0},
                completion_type="chat",
                processor_func=mock_processor
            )
            request_ids.append(request_id)
        
        # Wait for all results
        results = []
        for request_id in request_ids:
            result = await queue.wait_for_result(request_id, timeout=10.0)
            results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✓ Processed {len(results)} requests in {processing_time:.2f}s")
        print(f"✓ Expected ~2s for 2 concurrent slots, actual: {processing_time:.2f}s")
        
        # Check final status
        status = queue.get_queue_status()
        print(f"✓ Final queue status: {status}")
        
    finally:
        await queue.stop()

async def test_queue_errors():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")
    
    queue = RequestQueue(max_concurrent_requests=1, max_queue_size=2)
    await queue.start()
    
    try:
        # Test failed request
        request_id = await queue.enqueue_request(
            data={"prompt": "Fail me", "should_fail": True},
            completion_type="chat",
            processor_func=mock_processor
        )
        
        try:
            await queue.wait_for_result(request_id, timeout=5.0)
            print("✗ Expected error but got result")
        except Exception as e:
            print(f"✓ Correctly handled error: {e}")
        
        # Test queue overflow
        request_ids = []
        for i in range(3):  # More than max_queue_size
            try:
                request_id = await queue.enqueue_request(
                    data={"prompt": f"Request {i}", "processing_time": 2.0},
                    completion_type="chat",
                    processor_func=mock_processor
                )
                request_ids.append(request_id)
            except Exception as e:
                print(f"✓ Queue overflow handled: {e}")
                break
        
        # Clean up pending requests
        for request_id in request_ids:
            try:
                await asyncio.wait_for(queue.wait_for_result(request_id), timeout=3.0)
            except:
                pass
        
    finally:
        await queue.stop()

async def test_queue_timeout():
    """Test timeout handling."""
    print("\n=== Testing Timeout Handling ===")
    
    queue = RequestQueue(max_concurrent_requests=1, max_queue_size=5)
    await queue.start()
    
    try:
        # Submit a long-running request
        request_id = await queue.enqueue_request(
            data={"prompt": "Long task", "processing_time": 3.0},
            completion_type="chat",
            processor_func=mock_processor
        )
        
        try:
            # Try to get result with short timeout
            result = await queue.wait_for_result(request_id, timeout=1.0)
            print("✗ Expected timeout but got result")
        except asyncio.TimeoutError:
            print("✓ Timeout correctly handled")
        
    finally:
        await queue.stop()

async def main():
    """Run all tests."""
    print("Starting Request Queue Tests...")
    
    try:
        await test_queue_basic()
        await test_queue_concurrent()
        await test_queue_errors()
        await test_queue_timeout()
        print("\n🎉 All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
