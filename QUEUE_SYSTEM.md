# Request Queue System

The ezlocalai server now includes a robust request queuing system to handle concurrent chat completion requests without errors. This system ensures that multiple requests are processed efficiently and reliably.

## Features

- **Concurrent Request Handling**: Process multiple requests simultaneously (configurable)
- **Request Queuing**: Queue requests when system is at capacity
- **Timeout Management**: Automatic timeout handling for stuck requests
- **Status Monitoring**: Real-time queue status and metrics
- **Error Handling**: Graceful error handling with appropriate HTTP status codes

## Configuration

The queue system is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | 1 | Maximum requests processed simultaneously |
| `MAX_QUEUE_SIZE` | 100 | Maximum requests that can be queued |
| `REQUEST_TIMEOUT` | 300 | Request timeout in seconds |

### Example Configurations

Copy `queue-config.env.example` to `.env` or set these in your environment:

```bash
# For high-performance setups with sufficient GPU memory
MAX_CONCURRENT_REQUESTS=3
MAX_QUEUE_SIZE=50
REQUEST_TIMEOUT=180

# For memory-constrained environments
MAX_CONCURRENT_REQUESTS=1
MAX_QUEUE_SIZE=200
REQUEST_TIMEOUT=600
```

## API Endpoints

### Queue Management

#### Get Queue Status
```
GET /v1/queue/status
```

Returns current queue metrics:
```json
{
  "queue_size": 5,
  "processing_count": 1,
  "max_concurrent": 1,
  "max_queue_size": 100,
  "total_processed": 142,
  "total_failed": 3,
  "total_queued": 150,
  "active_requests": 6,
  "is_running": true
}
```

#### Get Request Status
```
GET /v1/queue/request/{request_id}
```

Returns status of a specific request:
```json
{
  "id": "12345678-1234-5678-9abc-123456789abc",
  "status": "processing",
  "completion_type": "chat",
  "timestamp": 1625097600.0,
  "error": null
}
```

## HTTP Status Codes

The queue system returns appropriate HTTP status codes:

- `200 OK`: Request completed successfully
- `408 Request Timeout`: Request exceeded timeout limit
- `503 Service Unavailable`: Queue is full, try again later
- `500 Internal Server Error`: Processing error occurred

## Monitoring and Troubleshooting

### Queue Metrics

Monitor the queue status endpoint to understand system load:
- `queue_size`: Requests waiting to be processed
- `processing_count`: Requests currently being processed
- `total_processed` / `total_failed`: Success/failure rates

### Log Messages

The queue system provides detailed logging:
- `[RequestQueue]`: Queue management messages
- `[Chat Completions]`: Endpoint-specific messages
- `[Completions]`: Completion-specific messages

### Common Issues

1. **Queue Full (503 errors)**
   - Increase `MAX_QUEUE_SIZE`
   - Scale up processing capacity
   - Implement client-side retry logic

2. **Timeout Errors (408)**
   - Increase `REQUEST_TIMEOUT`
   - Check for model loading issues
   - Monitor GPU memory usage

3. **High Memory Usage**
   - Decrease `MAX_CONCURRENT_REQUESTS`
   - Decrease `MAX_QUEUE_SIZE`
   - Monitor system resources

## Client Integration

### Python Example
```python
import openai
import time

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8091/v1"
)

# With automatic retry on queue full
def chat_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="your-model",
                messages=messages
            )
            return response
        except openai.APIError as e:
            if e.status_code == 503 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
```

### cURL Example
```bash
# Check queue status
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8091/v1/queue/status

# Send chat completion
curl -X POST \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"model":"your-model","messages":[{"role":"user","content":"Hello"}]}' \
     http://localhost:8091/v1/chat/completions
```

## Performance Tuning

### GPU Memory Optimization

1. **Single Model Instance**: Each concurrent request shares the same model instance
2. **Memory Monitoring**: Monitor GPU memory usage when increasing concurrency
3. **Batch Size**: Adjust `LLM_BATCH_SIZE` for optimal throughput

### Recommended Settings by Hardware

| GPU Memory | MAX_CONCURRENT_REQUESTS | MAX_QUEUE_SIZE |
|------------|-------------------------|----------------|
| 8GB        | 1                       | 50             |
| 16GB       | 2                       | 100            |
| 24GB+      | 3-4                     | 200            |

### Load Testing

Test your configuration under load:
```bash
# Simple load test with multiple concurrent requests
for i in {1..10}; do
  curl -X POST -H "Authorization: Bearer your-api-key" \
       -H "Content-Type: application/json" \
       -d '{"model":"your-model","messages":[{"role":"user","content":"Test"}]}' \
       http://localhost:8091/v1/chat/completions &
done
wait
```
