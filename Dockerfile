ARG LLAMACPP_IMAGE="full-cuda"
ARG BASE_IMAGE="ghcr.io/ggerganov/llama.cpp:${LLAMACPP_IMAGE}"
ARG MODEL_URL="${MODEL_URL}"
ARG QUANT_TYPE="${QUANT_TYPE}"
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN python3 GetModel.py "${MODEL_URL}" "${QUANT_TYPE}"
EXPOSE 8091
ENTRYPOINT ["/app/entrypoint.sh"]
