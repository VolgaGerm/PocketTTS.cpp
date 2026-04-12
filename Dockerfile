ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS exporter

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY export_onnx.py ./

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    python -m pip install \
        "pocket-tts @ git+https://github.com/kyutai-labs/pocket-tts.git" \
        onnx \
        onnxruntime

RUN mkdir -p /opt/pocket-tts/models && \
    python export_onnx.py --output-dir /opt/pocket-tts/models --no-validate


FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /src

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip cmake

COPY CMakeLists.txt pocket_tts.cpp ./

RUN cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIB=OFF && \
    cmake --build /tmp/build -j"$(nproc)"

RUN mkdir -p /opt/pocket-tts/runtime && \
    install -Dm755 /src/pocket-tts /opt/pocket-tts/runtime/pocket-tts && \
    cp -a /tmp/build/_deps/onnxruntime-src/lib/. /opt/pocket-tts/runtime/


FROM ${PYTHON_IMAGE} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV LD_LIBRARY_PATH=/app

COPY --from=builder /opt/pocket-tts/runtime/ /app/
COPY --from=exporter /opt/pocket-tts/models/ /app/models/

RUN mkdir -p /app/voices/.cache

EXPOSE 8080

ENTRYPOINT ["/app/pocket-tts"]
CMD ["--server", "--port", "8080"]
