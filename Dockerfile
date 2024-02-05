FROM python:3.10-bullseye
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip --no-cache-dir
WORKDIR /app
ENV HOST 0.0.0.0
COPY download.py .
RUN --mount=type=cache,target=/var/cache/models,sharing=shared \
    python3 download.py
COPY requirements.txt .
RUN --mount=type=cache,target=/var/cache/pip,sharing=locked \
    python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8091
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1", "--proxy-headers"]
