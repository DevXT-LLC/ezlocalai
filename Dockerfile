ARG LLAMACPP_IMAGE="full"
FROM "ghcr.io/ggerganov/llama.cpp:${LLAMACPP_IMAGE}"
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
ARG MODEL_URL="None"
RUN python3 provider.py --model_name ${MODEL_URL}
EXPOSE 8091
ENTRYPOINT ["/app/entrypoint.sh"]
