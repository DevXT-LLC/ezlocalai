FROM python:3.10-bullseye
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip --no-cache-dir
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HOST 0.0.0.0
EXPOSE 8091
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1", "--proxy-headers"]
