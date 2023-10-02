ARG LLAMACPP_IMAGE="full"
ARG BASE_IMAGE="ghcr.io/ggerganov/llama.cpp:${LLAMACPP_IMAGE}"
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
ARG MODEL_URL="${MODEL_URL}"
ARG QUANT_TYPE="${QUANT_TYPE}"
RUN python3 GetModel.py "${MODEL_URL}" "${QUANT_TYPE}"
EXPOSE 8091
ENTRYPOINT ["/app/entrypoint.sh"]
