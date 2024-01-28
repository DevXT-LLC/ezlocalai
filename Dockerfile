FROM python:3.10-bullseye
RUN apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV HOST 0.0.0.0
COPY . .
RUN python3 -m pip install --upgrade pip --no-cache-dir && \
    python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 local_llm/CTTS.py && python3 local_llm/STT.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]