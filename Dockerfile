ARG LLAMACPP_IMAGE="full"
FROM "ghcr.io/ggerganov/llama.cpp:${LLAMACPP_IMAGE}"
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8091
ENTRYPOINT ["/app/entrypoint.sh"]
