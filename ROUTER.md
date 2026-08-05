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