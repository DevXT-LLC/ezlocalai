# ezLocalai Router

The ezLocalai router acts as a load balancer for your running AI models across multiple machines, which we refer to as `ezLocalai workers`. Requests are sent to the router, it picks a worker to send the request to and acts as a middle man for load balancing and analytics. This enables you to scale easily. You can use your local hardware, spin up a VM with GPU on a cloud service like runpod to use as a worker, or open up our [Google Colab Notebook](https://colab.research.google.com/github/DevXT-LLC/ezlocalai/blob/main/ezlocalai-colab.ipynb) and point it at your router.

If you want your ezLocalai router to be accessible over the web, we recommend [setting up a Cloudflare tunnel](https://developers.cloudflare.com/tunnel/setup/) that points the domain/subdomain to port 8092 for the router.

## Prerequisites

- [Git](https://git-scm.com/)
- [Docker](https://docs.docker.com/get-docker/)

## Router Setup

Start by cloning the ezLocalai repository:

```bash
git clone https://github.com/devxt-llc/ezlocalai
cd ezlocalai
touch .env
```

**Optional but recommended**: If you want to set an API key, open the `.env` file and set `EZLOCALAI_API_KEY` to the desired value.

```bash
EZLOCALAI_API_KEY="The API key that you want to use to access the router"
```

To add Chutes as automatic text/vision overflow, set its API key. The model
defaults to `Qwen/Qwen3.8-27B-TEE` when omitted:

```bash
CHUTES_API_KEY="cpk_your-chutes-key"
# CHUTES_MODEL="Qwen/Qwen3.8-27B-TEE"
# CHUTES_MODEL="Qwen/Qwen3.8-27B-TEE,another/model"
```

The router exposes Chutes as a persistent t45 worker on the dashboard and
tracks its requests and input/output tokens with the same usage accounting as
local workers. Internal t45-or-faster workers win while available; Chutes is
used when they are occupied. The worker row caches the remaining Chutes USD
balance, seeded at router startup and refreshed after successful Chutes
inference requests. Its model is advertised with 100 concurrent slots. Send
the OpenRouter key to add a final tier-39 pool with 1,000 slots:

```bash
OPENROUTER_API_KEY="sk-or-v1-your-key"
# OPENROUTER_MODEL="qwen/qwen3.8-27b"
# OPENROUTER_MODEL="qwen/qwen3.8-27b,another/model"
```

OpenRouter receives traffic after internal t45-or-faster GPUs and Chutes. Its
remaining credits, requests, and tokens use the same dashboard accounting, and
its Qwen model shares the `Qwen3.8-27B` grouping. Send
`"disable_fallback": true` in a `/v1/chat/completions` body to exclude both
managed providers and wait for internal resources without a router-side
timeout.

Both model variables accept deduplicated comma-separated lists. The router
advertises each model, forwards the exact matching provider model ID, and keeps
one shared provider capacity pool (100 for Chutes, 1,000 for OpenRouter).
Chat-generation settings are preserved across fallback. Qwen3.8 receives the
same local thinking/instruct sampling profile, with `chat_template_kwargs`,
`reasoning`, and `reasoning_effort` translated to the provider-native thinking
control. Token limits, streaming, stop/seed, tools, tool choice, structured
outputs, and other supported sampling options pass through unchanged.

Then run the router server

```bash
docker compose -f docker-compose-router.yml pull && docker compose -f docker-compose-router.yml up -d
```

## Connecting ezLocalai Workers

Edit the `.env` of the ezLocalai worker to connect it to your ezLocalai router.

If the ezLocalai worker is not on your local network, it will need to go through a tunnel. We will default assume that it is not on the local network, change `WORKER_TUNNEL` to `false` if it is on the same network as the router.

```bash
ROUTER_URL=https://your.router.url
ROUTER_API_KEY="The API key that you want to use to access the router"
WORKER_TUNNEL=true
```
