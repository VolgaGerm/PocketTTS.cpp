# PocketTTS.cpp

Single-file C++ inference runtime for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), Kyutai's lightweight text-to-speech model. Runs entirely on CPU via ONNX Runtime with zero-shot voice cloning from short audio samples.

One file (`pocket_tts.cpp`), no frameworks, no Python dependency at runtime.

## Features

- **Single-file implementation** — all inference logic in one C++ source file
- **CLI, HTTP server, and shared library** — built from the same source
- **Streaming output** — chunked audio generation with low first-chunk latency (~96ms)
- **Voice cloning** — clone any voice from a short WAV sample
- **Two-layer disk cache** — voice embeddings (`.emb`) and transformer KV state (`.kv`) are cached to disk, making repeated use of the same voice near-instant
- **INT8 / FP32 precision** — INT8 by default for ~4x smaller models at comparable quality
- **Built-in profiler** — `--profile` flag for per-operation timing

## Requirements

- CMake 3.28+
- C++17 compiler (GCC, Clang)
- Linux or macOS (Windows support in CMakeLists.txt but untested)

All dependencies (ONNX Runtime, SentencePiece, dr_wav) are fetched automatically by CMake.

## Setup

Download the ONNX models from [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx) on HuggingFace and create the required directory structure:

```
PocketTTS.cpp/
├── CMakeLists.txt
├── pocket_tts.cpp
├── models/
│   ├── flow_lm_flow_int8.onnx
│   ├── flow_lm_flow.onnx
│   ├── flow_lm_main_int8.onnx
│   ├── flow_lm_main.onnx
│   ├── mimi_decoder_int8.onnx
│   ├── mimi_decoder.onnx
│   ├── mimi_encoder.onnx
│   ├── text_conditioner.onnx
│   └── tokenizer.model
└── voices/
    └── YourVoice.wav
```

Both `models/` and `voices/` must exist before running. Place at least one `.wav` voice sample in `voices/`.

For INT8 inference (default), you need the `_int8` variants plus `mimi_encoder.onnx` and `text_conditioner.onnx`. FP32 variants are optional unless you pass `--precision fp32`.

## Build

```bash
cmake -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build -j$(nproc)
```

This produces the `pocket-tts` CLI executable. To also build the shared library for FFI:

```bash
cmake -B .build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIB=ON
cmake --build .build -j$(nproc)
```

## Usage

### CLI

```bash
# Generate speech to a WAV file
./pocket-tts "Hello, world." voice.wav output.wav

# Pipe raw PCM to another process (e.g. aplay, ffplay, sox)
./pocket-tts --stdout "Hello, world." voice.wav | aplay -f FLOAT_LE -r 24000 -c 1

# INT8 (default) or FP32
./pocket-tts --precision fp32 "Hello." voice.wav output.wav

# Adjust generation parameters
./pocket-tts --temperature 0.5 --lsd-steps 10 "Hello." voice.wav output.wav
```

The voice argument can be a filename in the `voices/` directory (e.g. `voice.wav`) or an absolute path to any WAV file.

### HTTP Server

```bash
./pocket-tts --server --port 8080
```

Endpoints:
- `POST /tts` — generate speech (JSON body: `{"text": "...", "voice": "..."}`)
- `GET /health` — health check

Audio is streamed back as chunked `audio/pcm;rate=24000;encoding=float;bits=32`.

### Shared Library (FFI)

Build with `-DBUILD_SHARED_LIB=ON` to produce `libpocket_tts.so`. The C API:

```c
void*  ptt_create(const char* models_dir, const char* voices_dir,
                  const char* tokenizer_path, const char* precision,
                  float temperature, int lsd_steps, int num_threads);
double ptt_warmup(void* handle);
int    ptt_synthesize(void* handle, const char* text, const char* voice,
                      float** out_samples, int* out_len, int* out_sample_rate);
void   ptt_free_audio(float* samples);
void   ptt_destroy(void* handle);

// Streaming
void*  ptt_stream_start(void* handle, const char* text, const char* voice);
int    ptt_stream_read(void* stream_ctx, float** out_samples, int* out_len);
void   ptt_stream_end(void* stream_ctx);
```

## Caching

PocketTTS uses two layers of disk caching, both stored under `voices/.cache/`:

**Voice embeddings (`.emb`)** — The output of the Mimi encoder for each voice sample. Avoids re-encoding the same WAV file on every run. Generated automatically on first use.

**KV state snapshots (`.kv`)** — The transformer's internal KV cache state after voice conditioning. This is the expensive part — on a cold start, voice conditioning takes hundreds of milliseconds. A cached `.kv` file restores in ~4ms. For multi-sentence input, the KV snapshot is also held in memory so only the first sentence pays the disk load cost.

Cache files are invalidated automatically when the source WAV is modified. To clear all caches:

```bash
rm -rf voices/.cache/
```

To disable caching entirely, pass `--no-cache`.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--precision` | `int8` | Model precision (`int8` or `fp32`) |
| `--temperature` | `0.7` | Sampling temperature |
| `--lsd-steps` | `1` | Flow matching ODE solver steps |
| `--threads` | `0` | Thread count (`0` = half of available cores) |
| `--models-dir` | `models` | Path to ONNX model directory |
| `--voices-dir` | `voices` | Path to voice samples directory |
| `--no-cache` | — | Disable all disk caching (`.emb` and `.kv` files) |
| `--stdout` | — | Output raw f32le PCM to stdout |
| `--profile` | — | Print per-operation timing report after generation |
| `--server` | — | Start HTTP server mode |
| `--port` | `8080` | Server port |

## Acknowledgments

- [Kyutai Labs](https://github.com/kyutai-labs/pocket-tts) — Pocket TTS model and original Python implementation (MIT)
- [KevinAHM](https://github.com/KevinAHM/pocket-tts-onnx-export) — ONNX model export pipeline and INT8 quantization (MIT)

## License

MIT — see [LICENSE](LICENSE).
