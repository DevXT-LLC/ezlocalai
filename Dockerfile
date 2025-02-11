FROM python:3.10-bullseye
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget libsndfile1 cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip --no-cache-dir
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/Josh-XT/DeepSeek-VL deepseek && \
    cd deepseek && \
    pip install --no-cache-dir -e . && \
    cd .. && \
    pip uninstall -y torchaudio && \
    pip install --no-cache-dir torchaudio==2.3.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install diffusers --no-cache-dir && \
    pip install spacy==3.7.4 && \
    python -m spacy download en_core_web_sm
COPY . .
ENV HOST 0.0.0.0
EXPOSE 8091
EXPOSE 8502
CMD streamlit run ui.py & uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers