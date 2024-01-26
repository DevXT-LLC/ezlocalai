FROM ubuntu:22.04
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev libopenblas-dev && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV HOST 0.0.0.0
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 local_llm/CTTS.py
RUN python3 local_llm/STT.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]