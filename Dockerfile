FROM python:3.10-bullseye
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip --no-cache-dir
WORKDIR /app

# Install PyTorch CPU version
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install numpy and Cython for pkuseg
RUN pip install numpy==1.25.2 Cython --no-cache-dir

# Install pkuseg with build isolation disabled
RUN pip install pkuseg==0.0.25 --no-build-isolation --no-cache-dir

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    pip install spacy spacy-legacy spacy-loggers && \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Install xllamacpp CPU version  
RUN pip install xllamacpp --force-reinstall --no-cache-dir

COPY . .
ENV HOST 0.0.0.0 \
    TOKENIZERS_PARALLELISM=false
EXPOSE 8091
EXPOSE 8502
CMD streamlit run ui.py & uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers