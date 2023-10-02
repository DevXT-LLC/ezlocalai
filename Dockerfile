ARG LLAMACPP_IMAGE="full-cuda"
ARG BASE_IMAGE="ghcr.io/ggerganov/llama.cpp:${LLAMACPP_IMAGE}"
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8091
ENTRYPOINT ["/app/entrypoint.sh"]
