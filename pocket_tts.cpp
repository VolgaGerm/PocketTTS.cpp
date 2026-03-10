// PocketTTS.cpp — Single-file C++ TTS runtime using ONNX Runtime
// https://github.com/VolgaGerm/PocketTTS.cpp
//
// Build with CMake:
//   cmake -B .build -DCMAKE_BUILD_TYPE=Release
//   cmake --build .build -j$(nproc)

// ── External Libraries ──────────────────────────────────────────────────────

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// ── Standard Library ────────────────────────────────────────────────────────

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ── POSIX ───────────────────────────────────────────────────────────────────

#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>

namespace pocket_tts {

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

static size_t calc_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) n *= (d > 0 ? d : 1);
    return n;
}

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    
    Tensor() = default;
    Tensor(std::vector<int64_t> s) : shape(std::move(s)), data(calc_numel(shape), 0.0f) {}
    Tensor(std::vector<float> d, std::vector<int64_t> s) : shape(std::move(s)), data(std::move(d)) {}
    
    size_t numel() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    
    Tensor& reshape(std::vector<int64_t> ns) {
        int64_t neg = -1, known = 1;
        for (size_t i = 0; i < ns.size(); ++i) {
            if (ns[i] == -1) neg = i;
            else known *= ns[i];
        }
        if (neg >= 0) ns[neg] = numel() / known;
        shape = std::move(ns);
        return *this;
    }
    
    Tensor squeeze(int64_t dim = -1) const {
        std::vector<int64_t> ns;
        for (size_t i = 0; i < shape.size(); ++i)
            if (shape[i] != 1 || (dim >= 0 && (int64_t)i != dim)) ns.push_back(shape[i]);
        if (ns.empty()) ns.push_back(1);
        return Tensor(data, ns);
    }
    
    static Tensor concat(const std::vector<Tensor>& ts, int64_t dim) {
        if (ts.empty()) throw std::runtime_error("Cannot concat empty list");
        if (dim < 0) dim += ts[0].shape.size();
        
        std::vector<int64_t> os = ts[0].shape;
        int64_t total = 0;
        for (const auto& t : ts) total += t.shape[dim];
        os[dim] = total;
        
        Tensor r(os);
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= os[i];
        for (size_t i = dim + 1; i < os.size(); ++i) inner *= os[i];
        
        int64_t off = 0;
        for (const auto& t : ts) {
            int64_t td = t.shape[dim], chunk = td * inner;
            for (int64_t o = 0; o < outer; ++o)
                std::memcpy(r.data.data() + o * total * inner + off * inner,
                           t.data.data() + o * chunk, chunk * sizeof(float));
            off += td;
        }
        return r;
    }
    
};

struct TensorI64 {
    std::vector<int64_t> shape;
    std::vector<int64_t> data;
    TensorI64() = default;
    TensorI64(std::vector<int64_t> s) : shape(std::move(s)), data(calc_numel(shape), 0) {}
    size_t numel() const { return data.size(); }
    int64_t* ptr() { return data.data(); }
    const int64_t* ptr() const { return data.data(); }
};

struct Config {
    std::string models_dir = "models", tokenizer_path = "models/tokenizer.model";
    std::string voices_dir = "voices";
    std::string precision = "int8";
    float temperature = 0.7f;
    float eos_threshold = -4.0f;
    float noise_clamp = 3.0f;
    int lsd_steps = 1, num_threads = 0, first_chunk_frames = 1, max_chunk_frames = 15;
    int eos_extra_frames = 10;
    bool verbose = false;
    bool voice_cache = true;
};

struct AudioData {
    std::vector<float> samples;
    int sample_rate = 24000;
    float duration_sec() const { return float(samples.size()) / sample_rate; }
};

using StreamCallback = std::function<bool(const float*, size_t)>;

// ════════════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════════════

// ── RNG (xoshiro256**) ──────────────────────────────────────────────────────

namespace rng {
static uint64_t s[4] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t next() {
    uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

void seed(uint64_t v) {
    for (int i = 0; i < 4; ++i) {
        v += 0x9E3779B97F4A7C15ULL;
        uint64_t z = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[i] = z ^ (z >> 31);
    }
}

static float uniform() { return (next() >> 11) * (1.0f / 9007199254740992.0f); }

float normal(float mean = 0, float stddev = 1) {
    float u1 = uniform(), u2 = uniform();
    while (u1 <= 1e-10f) u1 = uniform();
    return mean + stddev * std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307179586f * u2);
}

void fill_normal(float* data, size_t n, float mean = 0, float stddev = 1) {
    for (size_t i = 0; i < n; ++i) data[i] = normal(mean, stddev);
}
} // namespace rng

// ── Audio Resampling (Lanczos) ──────────────────────────────────────────────

static std::vector<float> resample(const std::vector<float>& in, int src, int dst) {
    if (src == dst) return in;
    constexpr int K = 16;
    constexpr float PI = 3.14159265358979323846f;
    double ratio = double(dst) / src;
    std::vector<float> out(size_t(in.size() * ratio));
    
    auto sinc = [](float x) { return std::abs(x) < 1e-6f ? 1.0f : std::sin(PI * x) / (PI * x); };
    auto lanczos = [&](float x) { return std::abs(x) >= K ? 0.0f : sinc(x) * sinc(x / K); };
    
    for (size_t i = 0; i < out.size(); ++i) {
        double sp = i / ratio;
        int64_t c = int64_t(sp);
        float f = float(sp - c), sample = 0, wsum = 0;
        for (int k = -K + 1; k <= K; ++k) {
            int64_t idx = c + k;
            if (idx >= 0 && idx < int64_t(in.size())) {
                float w = lanczos(k - f);
                sample += in[idx] * w;
                wsum += w;
            }
        }
        out[i] = wsum > 0 ? sample / wsum : 0;
    }
    return out;
}

// ── Sentence Splitting ──────────────────────────────────────────────────────
// Splits on sentence-ending punctuation (. ! ?) followed by whitespace or EOF.
// Preserves punctuation with the sentence. Handles common abbreviations.

static std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    
    auto is_abbreviation = [](const std::string& s, size_t dot_pos) -> bool {
        if (dot_pos < 2) return false;
        size_t start = dot_pos;
        while (start > 0 && std::isalpha((unsigned char)s[start - 1])) start--;
        std::string word = s.substr(start, dot_pos - start);
        for (auto& c : word) c = std::tolower((unsigned char)c);
        return word == "mr" || word == "mrs" || word == "ms" || word == "dr" || 
               word == "st" || word == "jr" || word == "sr" || word == "vs" ||
               word == "etc" || word == "inc" || word == "ltd" || word == "prof" ||
               word == "gen" || word == "gov" || word == "sgt" || word == "cpl" ||
               word == "pvt" || word == "capt" || word == "lt" || word == "col";
    };
    
    for (size_t i = 0; i < text.size(); ++i) {
        current += text[i];
        
        if ((text[i] == '.' || text[i] == '!' || text[i] == '?')) {
            if (text[i] == '.' && i + 1 < text.size() && text[i + 1] == '.') continue;
            if (text[i] == '.' && i > 0 && text[i - 1] == '.') continue;
            if (text[i] == '.' && is_abbreviation(text, i)) continue;
            if (i + 1 >= text.size() || text[i + 1] == ' ' || text[i + 1] == '"' || text[i + 1] == '\'') {
                size_t start = current.find_first_not_of(" \t\n\r");
                if (start != std::string::npos) {
                    sentences.push_back(current.substr(start));
                }
                current.clear();
            }
        }
    }
    
    if (!current.empty()) {
        size_t start = current.find_first_not_of(" \t\n\r");
        if (start != std::string::npos) {
            sentences.push_back(current.substr(start));
        }
    }
    
    return sentences;
}

// ════════════════════════════════════════════════════════════════════════════
// Profiler
// ════════════════════════════════════════════════════════════════════════════

struct Profiler {
    struct Timer {
        std::string name;
        double total_ms = 0;
        int count = 0;
        double min_ms = 1e9, max_ms = 0;
        
        void add(double ms) {
            total_ms += ms;
            count++;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }
        double avg_ms() const { return count > 0 ? total_ms / count : 0; }
    };
    
    std::unordered_map<std::string, Timer> timers;
    bool enabled = false;
    
    class ScopedTimer {
        Profiler& prof;
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
    public:
        ScopedTimer(Profiler& p, const std::string& n) : prof(p), name(n), start(std::chrono::high_resolution_clock::now()) {}
        ~ScopedTimer() {
            if (prof.enabled) {
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                prof.timers[name].name = name;
                prof.timers[name].add(ms);
            }
        }
    };
    
    ScopedTimer time(const std::string& name) { return ScopedTimer(*this, name); }
    
    void report() const {
        std::cout << "\n========== PROFILING REPORT ==========\n";
        std::vector<std::pair<std::string, Timer>> sorted;
        for (const auto& [k, v] : timers) sorted.emplace_back(k, v);
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::left << std::setw(35) << "Operation" 
                  << std::right << std::setw(10) << "Total(ms)"
                  << std::setw(10) << "Count"
                  << std::setw(10) << "Avg(ms)"
                  << std::setw(10) << "Min(ms)"
                  << std::setw(10) << "Max(ms)" << "\n";
        std::cout << std::string(85, '-') << "\n";
        
        for (const auto& [name, t] : sorted) {
            std::cout << std::left << std::setw(35) << name
                      << std::right << std::setw(10) << t.total_ms
                      << std::setw(10) << t.count
                      << std::setw(10) << t.avg_ms()
                      << std::setw(10) << t.min_ms
                      << std::setw(10) << t.max_ms << "\n";
        }
        std::cout << "=======================================\n";
    }
    
    void reset() { timers.clear(); }
};

static Profiler g_prof;

// ════════════════════════════════════════════════════════════════════════════
// Disk Cache
//
// Two layers of on-disk caching, both stored under voices/.cache/:
//
//   .emb files — Mimi encoder output (voice embedding).
//                Avoids re-encoding the same WAV file on every run.
//
//   .kv files  — Transformer KV state after voice conditioning.
//                Avoids re-running the expensive voice conditioning pass.
//                On cache hit, restoring a KV snapshot takes ~4ms vs
//                hundreds of ms for a full conditioning pass.
// ════════════════════════════════════════════════════════════════════════════

namespace cache {

static time_t get_mtime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_mtime;
}

static bool mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        std::string sub = path.substr(0, pos);
        if (!sub.empty()) mkdir(sub.c_str(), 0755);
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// Derive cache file path: voices/.cache/{stem}.{ext}
// ext = "emb" for voice embeddings, "kv" for KV state snapshots
static std::string get_cache_path(const std::string& voices_dir, const std::string& voice_path, const char* ext = "emb") {
    std::string filename = voice_path;
    size_t slash = voice_path.find_last_of("/\\");
    if (slash != std::string::npos) filename = voice_path.substr(slash + 1);
    size_t dot = filename.rfind('.');
    if (dot != std::string::npos) filename = filename.substr(0, dot);
    return voices_dir + "/.cache/" + filename + "." + ext;
}

static bool is_cache_valid(const std::string& voice_path, const std::string& cache_path) {
    time_t voice_mtime = get_mtime(voice_path);
    time_t cache_mtime = get_mtime(cache_path);
    return cache_mtime > 0 && cache_mtime >= voice_mtime;
}

// ── Voice Embedding (.emb) Format ───────────────────────────────────────────
// [4B magic "EMB1"] [4B ndims] [ndims*8B shape] [numel*4B float data]

static constexpr uint32_t EMB_MAGIC = 0x31424D45; // "EMB1" little-endian

static bool save_embedding(const std::string& path, const std::vector<int64_t>& shape, const std::vector<float>& data) {
    size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        mkdir_p(path.substr(0, slash));
    }
    
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    
    uint32_t magic = EMB_MAGIC;
    int32_t ndims = static_cast<int32_t>(shape.size());
    
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&ndims), 4);
    f.write(reinterpret_cast<const char*>(shape.data()), ndims * sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    return f.good();
}

static bool load_embedding(const std::string& path, std::vector<int64_t>& shape, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    
    uint32_t magic;
    int32_t ndims;
    
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != EMB_MAGIC) return false;
    
    f.read(reinterpret_cast<char*>(&ndims), 4);
    if (ndims <= 0 || ndims > 10) return false;
    
    shape.resize(ndims);
    f.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(int64_t));
    
    size_t numel = 1;
    for (int32_t i = 0; i < ndims; ++i) {
        if (shape[i] <= 0) return false;
        numel *= shape[i];
    }
    
    data.resize(numel);
    f.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
    
    return f.good();
}

} // namespace cache

// ════════════════════════════════════════════════════════════════════════════
// ONNX Runtime Wrappers
// ════════════════════════════════════════════════════════════════════════════

static Ort::Env& get_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pocket_tts");
    return env;
}

// ── OrtSession ──────────────────────────────────────────────────────────────
// Thin wrapper around Ort::Session that caches input/output names and shapes.

class OrtSession {
    Ort::Session sess_;
    std::vector<std::string> in_names_, out_names_;
    std::vector<const char*> in_ptrs_, out_ptrs_;
    std::vector<std::vector<int64_t>> in_shapes_;
    std::vector<ONNXTensorElementDataType> in_types_;
    std::string name_;
    
public:
    OrtSession(Ort::Env& env, const std::string& path, const Ort::SessionOptions& opts, const std::string& name = "")
        : sess_(env, path.c_str(), opts), name_(name.empty() ? path : name) {
        Ort::AllocatorWithDefaultOptions alloc;
        
        size_t num_in = sess_.GetInputCount();
        for (size_t i = 0; i < num_in; ++i) {
            auto n = sess_.GetInputNameAllocated(i, alloc);
            in_names_.push_back(n.get());
            auto ti = sess_.GetInputTypeInfo(i);
            auto tsi = ti.GetTensorTypeAndShapeInfo();
            in_shapes_.push_back(tsi.GetShape());
            in_types_.push_back(tsi.GetElementType());
        }
        
        size_t num_out = sess_.GetOutputCount();
        for (size_t i = 0; i < num_out; ++i) {
            auto n = sess_.GetOutputNameAllocated(i, alloc);
            out_names_.push_back(n.get());
        }
        
        for (const auto& n : in_names_) in_ptrs_.push_back(n.c_str());
        for (const auto& n : out_names_) out_ptrs_.push_back(n.c_str());
    }
    
    Ort::Session& session() { return sess_; }
    
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& in) {
        auto _ = g_prof.time("run:" + name_);
        return sess_.Run(Ort::RunOptions{nullptr}, in_ptrs_.data(), in.data(), in.size(), out_ptrs_.data(), out_ptrs_.size());
    }
    
    void run_with_binding(Ort::IoBinding& binding) {
        auto _ = g_prof.time("run:" + name_);
        sess_.Run(Ort::RunOptions{nullptr}, binding);
    }
    
    void print_info() const {
        std::cout << "\n  Model: " << name_ << "\n";
        std::cout << "    Inputs (" << in_names_.size() << "):\n";
        for (size_t i = 0; i < in_names_.size(); ++i) {
            std::cout << "      [" << i << "] " << in_names_[i] << " : ";
            std::cout << type_str(in_types_[i]) << " ";
            print_shape(in_shapes_[i]);
            std::cout << "\n";
        }
        std::cout << "    Outputs (" << out_names_.size() << "):\n";
        for (size_t i = 0; i < out_names_.size(); ++i) {
            std::cout << "      [" << i << "] " << out_names_[i] << "\n";
        }
    }
    
    static std::string type_str(ONNXTensorElementDataType t) {
        switch (t) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
            default: return "type(" + std::to_string(t) + ")";
        }
    }
    
    static void print_shape(const std::vector<int64_t>& shape) {
        std::cout << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            if (shape[i] < 0) std::cout << "?";
            else std::cout << shape[i];
        }
        std::cout << "]";
    }
    
    const std::string& name() const { return name_; }
    const std::vector<std::string>& input_names() const { return in_names_; }
    const std::vector<std::string>& output_names() const { return out_names_; }
    const std::vector<std::vector<int64_t>>& input_shapes() const { return in_shapes_; }
    const std::vector<ONNXTensorElementDataType>& input_types() const { return in_types_; }
};

// ── StateBufferIO ───────────────────────────────────────────────────────────
// Manages the stateful inputs/outputs of the autoregressive transformer.
//
// The flow_lm_main model has ~60 state tensors (KV cache layers) that must
// be fed back as inputs on each step. This struct:
//
//   1. Double-buffers all state tensors so the output of step N becomes the
//      input of step N+1 without copying (just swap the buffer index).
//   2. Handles mixed types (float32, int64, bool) across state tensors.
//   3. Supports both fixed-size and dynamic-size states.
//   4. Provides Snapshot (fast in-memory) and DiskSnapshot (serialized blob)
//      for caching voice-conditioned KV state across runs.

struct StateBufferIO {
    std::vector<std::vector<float>> f32[2];
    std::vector<std::vector<int64_t>> i64[2];
    std::vector<std::vector<uint8_t>> b8[2];
    std::vector<std::vector<int64_t>> shapes;
    std::vector<ONNXTensorElementDataType> types;
    std::vector<std::string> names;
    std::vector<bool> is_dynamic;
    int current_buf = 0;
    
    void init(OrtSession& s) {
        const auto& in_names = s.input_names();
        const auto& in_shapes = s.input_shapes();
        const auto& in_types = s.input_types();
        
        for (size_t i = 0; i < in_names.size(); ++i) {
            if (in_names[i].find("state_") != 0) continue;
            names.push_back(in_names[i]);
            
            std::vector<int64_t> sh;
            bool dynamic = false;
            for (auto d : in_shapes[i]) {
                if (d <= 0) dynamic = true;
                sh.push_back(d > 0 ? d : 0);
            }
            shapes.push_back(sh);
            types.push_back(in_types[i]);
            is_dynamic.push_back(dynamic);
            
            size_t sz = 1;
            for (auto d : sh) sz *= (d > 0 ? d : 1);
            size_t alloc = dynamic ? 0 : sz;
            
            for (int b = 0; b < 2; ++b) {
                if (in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    i64[b].push_back(std::vector<int64_t>(alloc, 0));
                    f32[b].push_back({});
                    b8[b].push_back({});
                } else if (in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                    b8[b].push_back(std::vector<uint8_t>(alloc, 0));
                    f32[b].push_back({});
                    i64[b].push_back({});
                } else {
                    f32[b].push_back(std::vector<float>(alloc, 0.0f));
                    i64[b].push_back({});
                    b8[b].push_back({});
                }
            }
        }
    }
    
    int in_buf() const { return current_buf; }
    int out_buf() const { return 1 - current_buf; }
    void swap() { current_buf = 1 - current_buf; }
    
    Ort::Value create_input_value(size_t state_idx, Ort::MemoryInfo& mem) {
        int b = in_buf();
        auto t = types[state_idx];
        auto& sh = shapes[state_idx];
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return Ort::Value::CreateTensor<int64_t>(mem, i64[b][state_idx].data(), i64[b][state_idx].size(), 
                                                      sh.data(), sh.size());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            return Ort::Value::CreateTensor<bool>(mem, reinterpret_cast<bool*>(b8[b][state_idx].data()), 
                                                   b8[b][state_idx].size(), sh.data(), sh.size());
        } else {
            return Ort::Value::CreateTensor<float>(mem, f32[b][state_idx].data(), f32[b][state_idx].size(),
                                                    sh.data(), sh.size());
        }
    }
    
    Ort::Value create_output_value(size_t state_idx, Ort::MemoryInfo& mem) {
        int b = out_buf();
        auto t = types[state_idx];
        auto& sh = shapes[state_idx];
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return Ort::Value::CreateTensor<int64_t>(mem, i64[b][state_idx].data(), i64[b][state_idx].size(),
                                                      sh.data(), sh.size());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            return Ort::Value::CreateTensor<bool>(mem, reinterpret_cast<bool*>(b8[b][state_idx].data()),
                                                   b8[b][state_idx].size(), sh.data(), sh.size());
        } else {
            return Ort::Value::CreateTensor<float>(mem, f32[b][state_idx].data(), f32[b][state_idx].size(),
                                                    sh.data(), sh.size());
        }
    }
    
    void copy_from_output(size_t state_idx, Ort::Value& val) {
        int b = out_buf();
        auto info = val.GetTensorTypeAndShapeInfo();
        auto out_shape = info.GetShape();
        size_t out_size = info.GetElementCount();
        
        shapes[state_idx] = out_shape;
        
        auto t = types[state_idx];
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            i64[b][state_idx].resize(out_size);
            if (out_size > 0) memcpy(i64[b][state_idx].data(), val.GetTensorData<int64_t>(), out_size * sizeof(int64_t));
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            b8[b][state_idx].resize(out_size);
            if (out_size > 0) memcpy(b8[b][state_idx].data(), val.GetTensorData<bool>(), out_size);
        } else {
            f32[b][state_idx].resize(out_size);
            if (out_size > 0) memcpy(f32[b][state_idx].data(), val.GetTensorData<float>(), out_size * sizeof(float));
        }
    }
    
    // ── DiskSnapshot ────────────────────────────────────────────────────────
    // Serialized blob for persisting KV state to disk (.kv files).
    // Format: [4B current_buf] [4B num_states] then per-state:
    //         [4B ndims] [ndims*8B shape] [4B type] [8B data_bytes] [data]
    
    struct DiskSnapshot {
        std::vector<uint8_t> blob;
        static constexpr uint32_t MAGIC = 0x3143564B;  // "KVC1" little-endian
        
        bool save_to_disk(const std::string& path) const {
            size_t slash = path.find_last_of('/');
            if (slash != std::string::npos) cache::mkdir_p(path.substr(0, slash));
            std::ofstream f(path, std::ios::binary);
            if (!f) return false;
            uint32_t magic = MAGIC;
            uint64_t sz = blob.size();
            f.write(reinterpret_cast<const char*>(&magic), 4);
            f.write(reinterpret_cast<const char*>(&sz), 8);
            f.write(reinterpret_cast<const char*>(blob.data()), blob.size());
            return f.good();
        }
        
        bool load_from_disk(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) return false;
            uint32_t magic;
            uint64_t sz;
            f.read(reinterpret_cast<char*>(&magic), 4);
            if (magic != MAGIC) return false;
            f.read(reinterpret_cast<char*>(&sz), 8);
            if (sz == 0 || sz > 200 * 1024 * 1024) return false;
            blob.resize(sz);
            f.read(reinterpret_cast<char*>(blob.data()), sz);
            return f.good();
        }
    };
    
    // ── Snapshot ─────────────────────────────────────────────────────────────
    // Fast in-memory snapshot: all state data packed into contiguous buffers.
    // Restoring is a bulk memcpy into pre-sized buffers (~1ms for 60 states).
    
    struct Snapshot {
        std::vector<float> f32_data;
        std::vector<int64_t> i64_data;
        std::vector<uint8_t> b8_data;
        std::vector<size_t> f32_offsets;
        std::vector<size_t> i64_offsets;
        std::vector<size_t> b8_offsets;
        std::vector<std::vector<int64_t>> shapes;
        int current_buf;
    };
    
    Snapshot take_snapshot() const {
        Snapshot snap;
        int b = in_buf();
        size_t n = names.size();
        snap.shapes = shapes;
        snap.current_buf = current_buf;
        
        size_t total_f32 = 0, total_i64 = 0, total_b8 = 0;
        for (size_t i = 0; i < n; ++i) {
            total_f32 += f32[b][i].size();
            total_i64 += i64[b][i].size();
            total_b8 += b8[b][i].size();
        }
        
        snap.f32_data.resize(total_f32);
        snap.i64_data.resize(total_i64);
        snap.b8_data.resize(total_b8);
        snap.f32_offsets.resize(n + 1);
        snap.i64_offsets.resize(n + 1);
        snap.b8_offsets.resize(n + 1);
        
        size_t fo = 0, io = 0, bo = 0;
        for (size_t i = 0; i < n; ++i) {
            snap.f32_offsets[i] = fo;
            snap.i64_offsets[i] = io;
            snap.b8_offsets[i] = bo;
            if (!f32[b][i].empty()) { memcpy(snap.f32_data.data() + fo, f32[b][i].data(), f32[b][i].size() * sizeof(float)); fo += f32[b][i].size(); }
            if (!i64[b][i].empty()) { memcpy(snap.i64_data.data() + io, i64[b][i].data(), i64[b][i].size() * sizeof(int64_t)); io += i64[b][i].size(); }
            if (!b8[b][i].empty()) { memcpy(snap.b8_data.data() + bo, b8[b][i].data(), b8[b][i].size()); bo += b8[b][i].size(); }
        }
        snap.f32_offsets[n] = fo;
        snap.i64_offsets[n] = io;
        snap.b8_offsets[n] = bo;
        
        return snap;
    }
    
    void restore_snapshot(const Snapshot& snap) {
        current_buf = snap.current_buf;
        int b = in_buf();
        size_t n = names.size();
        shapes = snap.shapes;
        
        for (size_t i = 0; i < n; ++i) {
            size_t f32_count = snap.f32_offsets[i + 1] - snap.f32_offsets[i];
            size_t i64_count = snap.i64_offsets[i + 1] - snap.i64_offsets[i];
            size_t b8_count = snap.b8_offsets[i + 1] - snap.b8_offsets[i];
            
            f32[b][i].resize(f32_count);
            i64[b][i].resize(i64_count);
            b8[b][i].resize(b8_count);
            
            if (f32_count) memcpy(f32[b][i].data(), snap.f32_data.data() + snap.f32_offsets[i], f32_count * sizeof(float));
            if (i64_count) memcpy(i64[b][i].data(), snap.i64_data.data() + snap.i64_offsets[i], i64_count * sizeof(int64_t));
            if (b8_count) memcpy(b8[b][i].data(), snap.b8_data.data() + snap.b8_offsets[i], b8_count);
        }
    }
    
    DiskSnapshot snapshot_to_disk(const Snapshot& snap) const {
        DiskSnapshot ds;
        size_t n = names.size();
        
        size_t total = 8;
        for (size_t i = 0; i < n; ++i) {
            size_t f32_count = snap.f32_offsets[i + 1] - snap.f32_offsets[i];
            size_t i64_count = snap.i64_offsets[i + 1] - snap.i64_offsets[i];
            size_t b8_count = snap.b8_offsets[i + 1] - snap.b8_offsets[i];
            total += 4 + snap.shapes[i].size() * 8 + 4 + 8;
            if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) total += i64_count * 8;
            else if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) total += b8_count;
            else total += f32_count * 4;
        }
        
        ds.blob.resize(total);
        uint8_t* p = ds.blob.data();
        auto write = [&](const void* src, size_t bytes) { memcpy(p, src, bytes); p += bytes; };
        
        int32_t cb = snap.current_buf, ns = int32_t(n);
        write(&cb, 4); write(&ns, 4);
        
        for (size_t i = 0; i < n; ++i) {
            int32_t ndims = int32_t(snap.shapes[i].size());
            int32_t type = int32_t(types[i]);
            write(&ndims, 4);
            write(snap.shapes[i].data(), ndims * 8);
            write(&type, 4);
            
            size_t f32_count = snap.f32_offsets[i + 1] - snap.f32_offsets[i];
            size_t i64_count = snap.i64_offsets[i + 1] - snap.i64_offsets[i];
            size_t b8_count = snap.b8_offsets[i + 1] - snap.b8_offsets[i];
            
            int64_t data_bytes;
            if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                data_bytes = i64_count * 8; write(&data_bytes, 8);
                write(snap.i64_data.data() + snap.i64_offsets[i], data_bytes);
            } else if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                data_bytes = b8_count; write(&data_bytes, 8);
                write(snap.b8_data.data() + snap.b8_offsets[i], data_bytes);
            } else {
                data_bytes = f32_count * 4; write(&data_bytes, 8);
                write(snap.f32_data.data() + snap.f32_offsets[i], data_bytes);
            }
        }
        return ds;
    }
    
    void restore_from_disk(const DiskSnapshot& ds) {
        const uint8_t* p = ds.blob.data();
        auto read = [&](void* dst, size_t bytes) { memcpy(dst, p, bytes); p += bytes; };
        
        int32_t cb, ns;
        read(&cb, 4); read(&ns, 4);
        current_buf = cb;
        int b = in_buf();
        
        for (int32_t i = 0; i < ns; ++i) {
            int32_t ndims, type;
            int64_t data_bytes;
            read(&ndims, 4);
            shapes[i].resize(ndims);
            read(shapes[i].data(), ndims * 8);
            read(&type, 4);
            read(&data_bytes, 8);
            if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                size_t count = data_bytes / 8; i64[b][i].resize(count); read(i64[b][i].data(), data_bytes);
            } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                b8[b][i].resize(data_bytes); read(b8[b][i].data(), data_bytes);
            } else {
                size_t count = data_bytes / 4; f32[b][i].resize(count); read(f32[b][i].data(), data_bytes);
            }
        }
    }
};

// ── StatefulRunner ──────────────────────────────────────────────────────────
// Combines an OrtSession with a StateBufferIO and an IoBinding to run the
// autoregressive model efficiently. Non-state inputs are passed in per-step;
// state inputs/outputs are managed automatically via double-buffering.

class StatefulRunner {
    OrtSession& sess_;
    Ort::MemoryInfo mem_;
    StateBufferIO state_;
    std::unique_ptr<Ort::IoBinding> binding_;
    
public:
    StatefulRunner(OrtSession& sess) 
        : sess_(sess), mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        state_.init(sess);
        binding_ = std::make_unique<Ort::IoBinding>(sess_.session());
    }
    
    StateBufferIO& state() { return state_; }
    
    using Snapshot = StateBufferIO::Snapshot;
    using DiskSnapshot = StateBufferIO::DiskSnapshot;
    Snapshot take_snapshot() const { return state_.take_snapshot(); }
    void restore_snapshot(const Snapshot& snap) { state_.restore_snapshot(snap); }
    void restore_from_disk(const DiskSnapshot& ds) { state_.restore_from_disk(ds); }
    DiskSnapshot snapshot_to_disk(const Snapshot& snap) const { return state_.snapshot_to_disk(snap); }
    
    void reinit() {
        state_ = StateBufferIO();
        state_.init(sess_);
    }
    
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& non_state_inputs) {
        binding_->ClearBoundInputs();
        binding_->ClearBoundOutputs();
        
        const auto& in_names = sess_.input_names();
        const auto& out_names = sess_.output_names();
        
        size_t non_state_idx = 0;
        size_t state_idx = 0;
        for (size_t i = 0; i < in_names.size(); ++i) {
            if (in_names[i].find("state_") == 0) {
                binding_->BindInput(in_names[i].c_str(), state_.create_input_value(state_idx++, mem_));
            } else {
                binding_->BindInput(in_names[i].c_str(), non_state_inputs[non_state_idx++]);
            }
        }
        
        std::vector<std::pair<size_t, size_t>> dynamic_out_states;
        state_idx = 0;
        for (size_t i = 0; i < out_names.size(); ++i) {
            if (out_names[i].find("out_state_") == 0) {
                if (state_.is_dynamic[state_idx]) {
                    binding_->BindOutput(out_names[i].c_str(), mem_);
                    dynamic_out_states.push_back({i, state_idx});
                } else {
                    binding_->BindOutput(out_names[i].c_str(), state_.create_output_value(state_idx, mem_));
                }
                state_idx++;
            } else {
                binding_->BindOutput(out_names[i].c_str(), mem_);
            }
        }
        
        sess_.run_with_binding(*binding_);
        auto outputs = binding_->GetOutputValues();
        for (auto& [out_idx, st_idx] : dynamic_out_states) {
            state_.copy_from_output(st_idx, outputs[out_idx]);
        }
        state_.swap();
        
        std::vector<Ort::Value> result;
        for (size_t i = 0; i < out_names.size(); ++i) {
            if (out_names[i].find("out_state_") != 0) {
                result.push_back(std::move(outputs[i]));
            }
        }
        return result;
    }
    
    Ort::MemoryInfo& mem() { return mem_; }
};

// ════════════════════════════════════════════════════════════════════════════
// Tokenizer
// ════════════════════════════════════════════════════════════════════════════

class Tokenizer {
    sentencepiece::SentencePieceProcessor proc_;
public:
    explicit Tokenizer(const std::string& path) {
        auto s = proc_.Load(path);
        if (!s.ok()) throw std::runtime_error("Failed to load tokenizer: " + s.ToString());
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        proc_.Encode(text, &ids);
        return ids;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// TTS Engine
// ════════════════════════════════════════════════════════════════════════════

class PocketTTS {
public:
    static constexpr int SR = 24000;
    
    explicit PocketTTS(const Config& cfg = {}) : cfg_(cfg) {
        rng::seed(uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        tok_ = std::make_unique<Tokenizer>(cfg_.tokenizer_path);
        
        // Thread budget: --threads sets the total. During pipelined streaming,
        // the AR generator and Mimi decoder run simultaneously, so we split the
        // budget between them. Non-pipelined models (encoder, text conditioner)
        // get the full budget since they run alone.
        int cores = std::max(1, int(std::thread::hardware_concurrency()));
        int total = cfg_.num_threads ? cfg_.num_threads : std::max(2, cores / 2);
        int threads_ar = std::min(std::max(1, total / 3), 4);
        int threads_dec = std::max(1, total - threads_ar);
        int threads_full = total;
        
        auto make_opts = [](int threads) {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(threads);
            opts.SetInterOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            return opts;
        };
        
        auto opts_full = make_opts(threads_full);
        auto opts_ar = make_opts(threads_ar);
        auto opts_dec = make_opts(threads_dec);
        
        if (cfg_.verbose) {
            std::cout << "\n========== ONNX RUNTIME INFO ==========\n";
            std::cout << "  ORT Version: " << OrtGetApiBase()->GetVersionString() << "\n";
            std::cout << "  Thread budget: " << total << " (AR: " << threads_ar 
                      << ", decoder: " << threads_dec << ", full: " << threads_full << ")\n";
            auto providers = Ort::GetAvailableProviders();
            std::cout << "  Execution Providers: ";
            for (size_t i = 0; i < providers.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << providers[i];
            }
            std::cout << "\n================================\n";
        }
        
        auto& env = get_ort_env();
        std::string sfx = cfg_.precision == "int8" ? "_int8" : "";
        
        enc_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/mimi_encoder.onnx", opts_full, "mimi_encoder");
        txt_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/text_conditioner.onnx", opts_full, "text_conditioner");
        main_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/flow_lm_main" + sfx + ".onnx", opts_ar, "flow_lm_main" + sfx);
        flow_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/flow_lm_flow" + sfx + ".onnx", opts_ar, "flow_lm_flow" + sfx);
        dec_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/mimi_decoder" + sfx + ".onnx", opts_dec, "mimi_decoder" + sfx);
        
        main_runner_ = std::make_unique<StatefulRunner>(*main_);
        
        dt_ = 1.0f / cfg_.lsd_steps;
        st_values_.reserve(cfg_.lsd_steps);
        for (int j = 0; j < cfg_.lsd_steps; ++j) {
            float s = float(j) / cfg_.lsd_steps;
            st_values_.emplace_back(s, s + dt_);
        }
        
        if (cfg_.verbose) {
            std::cerr << "\n========== MODEL INFO ==========\n";
            main_->print_info();
            dec_->print_info();
            std::cerr << "================================\n";
        }
    }
    
    // ── Audio I/O ───────────────────────────────────────────────────────────
    
    static AudioData load_audio(const std::string& path) {
        auto _ = g_prof.time("load_audio");
        unsigned ch, sr;
        drwav_uint64 n;
        float* raw = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &n, nullptr);
        if (!raw) throw std::runtime_error("Failed to load: " + path);
        
        std::vector<float> mono(n);
        for (size_t i = 0; i < n; ++i) {
            float sum = 0;
            for (unsigned c = 0; c < ch; ++c) sum += raw[i * ch + c];
            mono[i] = sum / ch;
        }
        drwav_free(raw, nullptr);
        
        if (sr != SR) {
            auto _ = g_prof.time("resample");
            mono = resample(mono, sr, SR);
        }
        
        float mx = *std::max_element(mono.begin(), mono.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
        if (std::abs(mx) > 1) for (auto& s : mono) s /= std::abs(mx);
        
        return {std::move(mono), SR};
    }
    
    static void save_audio(const AudioData& a, const std::string& path) {
        auto _ = g_prof.time("save_audio");
        drwav w;
        drwav_data_format fmt{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, drwav_uint32(a.sample_rate), 32};
        if (!drwav_init_file_write(&w, path.c_str(), &fmt, nullptr))
            throw std::runtime_error("Failed to write: " + path);
        drwav_write_pcm_frames(&w, a.samples.size(), a.samples.data());
        drwav_uninit(&w);
    }
    
    // ── Voice Encoding ──────────────────────────────────────────────────────
    
    Tensor encode_voice(const std::string& path) {
        auto timer = g_prof.time("encode_voice");
        
        if (cfg_.voice_cache) {
            std::string cache_path = cache::get_cache_path(cfg_.voices_dir, path);
            
            if (cache::is_cache_valid(path, cache_path)) {
                auto cache_timer = g_prof.time("encode_voice.cache_load");
                std::vector<int64_t> shape;
                std::vector<float> data;
                if (cache::load_embedding(cache_path, shape, data)) {
                    if (cfg_.verbose) {
                        std::cerr << "  Loaded cached embedding: " << cache_path << "\n";
                    }
                    return Tensor(std::move(data), std::move(shape));
                }
            }
        }
        
        auto a = load_audio(path);
        
        Tensor t({1, 1, int64_t(a.samples.size())});
        std::copy(a.samples.begin(), a.samples.end(), t.data.begin());
        
        Ort::MemoryInfo m = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> in;
        in.push_back(Ort::Value::CreateTensor<float>(m, t.ptr(), t.numel(), t.shape.data(), t.shape.size()));
        
        auto out = enc_->run(in);
        auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t n = 1;
        for (auto d : sh) n *= d;
        
        Tensor r(std::vector<float>(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n),
                 std::vector<int64_t>(sh.begin(), sh.end()));
        
        while (r.shape.size() > 3) r = r.squeeze(0);
        if (r.shape.size() < 3) r.reshape({1, r.shape[0], r.shape[1]});
        
        if (cfg_.voice_cache) {
            std::string cache_path = cache::get_cache_path(cfg_.voices_dir, path);
            if (cache::save_embedding(cache_path, r.shape, r.data)) {
                if (cfg_.verbose) {
                    std::cerr << "  Saved embedding cache: " << cache_path << "\n";
                }
            }
        }
        
        return r;
    }
    
    // ── Public API ──────────────────────────────────────────────────────────
    
    AudioData generate(const std::string& text, const std::string& voice, int max_frames = 500) {
        return generate(text, get_voice(voice), max_frames);
    }
    
    AudioData generate(const std::string& text, const Tensor& voice, int max_frames = 500);
    void stream(const std::string& text, const std::string& voice, StreamCallback cb, int max_frames = 500);
    void stream(const std::string& text, const Tensor& voice, StreamCallback cb, int max_frames = 500);
    const Config& config() const { return cfg_; }
    
    double warmup() {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor dummy_voice({1, 8, 1024});
        std::fill(dummy_voice.data.begin(), dummy_voice.data.end(), 0.0f);
        stream("Hi.", dummy_voice, [](const float*, size_t) { return true; }, 1);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void print_profiling_report() const { g_prof.report(); }
    void reset_profiling() { g_prof.reset(); }

private:
    Config cfg_;
    std::unique_ptr<OrtSession> enc_, txt_, main_, flow_, dec_;
    std::unique_ptr<Tokenizer> tok_;
    std::unique_ptr<StatefulRunner> main_runner_;
    std::vector<std::pair<float, float>> st_values_;
    float dt_;
    std::unordered_map<std::string, Tensor> vcache_;
    
    // ── Voice Resolution ────────────────────────────────────────────────────
    
    std::string resolve_voice_path(const std::string& p) const {
        if (!p.empty() && p[0] == '/') return p;
        return cfg_.voices_dir + "/" + p;
    }
    
    Tensor get_voice(const std::string& p) {
        voice_kv_path_ = p;
        auto it = vcache_.find(p);
        if (it != vcache_.end()) return it->second;
        
        std::string resolved = resolve_voice_path(p);
        return vcache_[p] = encode_voice(resolved);
    }
    
    // ── Tokenization ────────────────────────────────────────────────────────
    
    TensorI64 tokenize(const std::string& text) {
        auto _ = g_prof.time("tokenize");
        std::string t = text;
        size_t s = t.find_first_not_of(" \t\n\r");
        size_t e = t.find_last_not_of(" \t\n\r");
        if (s == std::string::npos) throw std::runtime_error("Empty text");
        t = t.substr(s, e - s + 1);
        if (std::isalnum((unsigned char)t.back())) t += ".";
        if (!t.empty() && std::islower((unsigned char)t[0])) t[0] = std::toupper((unsigned char)t[0]);
        
        auto ids = tok_->encode(t);
        if (cfg_.verbose) std::cerr << "  Tokens: " << ids.size() << " from " << t.size() << " chars\n";
        TensorI64 r({1, int64_t(ids.size())});
        for (size_t i = 0; i < ids.size(); ++i) r.data[i] = ids[i];
        return r;
    }
    
    // ── LatentGen ───────────────────────────────────────────────────────────
    // Autoregressive latent generator. Each call to next() runs the main
    // transformer for one frame, then solves the flow matching ODE to produce
    // a 32-dim latent vector for the Mimi decoder.
    //
    // Construction has two paths:
    //   - Full: runs voice conditioning + text conditioning from scratch
    //   - Cached: restores a KV snapshot then runs text conditioning only
    
    class LatentGen {
        PocketTTS& tts;
        int max_, idx_ = 0, extra_ = 0;
        int eos_frame_ = -1;
        bool done_ = false, eos_ = false;
        float temp_;
        Ort::MemoryInfo m_;
        
        StatefulRunner& main_runner_;
        
        std::vector<float> fx_, cl_, cond_, temb_;
        std::vector<int64_t> csh_, tsh_;
        std::vector<Ort::Value> flow_inputs_;
        
        static constexpr int64_t curr_shape_[3] = {1, 1, 32};
        static constexpr int64_t empty_text_shape_[3] = {1, 0, 1024};
        static constexpr int64_t empty_seq_shape_[3] = {1, 0, 32};
        static constexpr int64_t s_shape_[2] = {1, 1};
        static constexpr int64_t x_shape_[2] = {1, 32};
        
        std::vector<float> s_buf_{1}, t_buf_{1};
        
        void cond_pass(const float* d, size_t sz, const std::vector<int64_t>& sh) {
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<float>(m_, nullptr, 0, empty_seq_shape_, 3));
            inputs.push_back(Ort::Value::CreateTensor<float>(m_, const_cast<float*>(d), sz, sh.data(), sh.size()));
            main_runner_.run(inputs);
        }
        
    public:
        using Snapshot = StatefulRunner::Snapshot;
        
        // Full path: voice conditioning → (optional snapshot) → text conditioning
        LatentGen(PocketTTS& t, const Tensor& v, const TensorI64& tid, int max, Snapshot* out_voice_snap = nullptr)
            : tts(t), max_(max), m_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
              fx_(32, 0), cl_(32, std::numeric_limits<float>::quiet_NaN()),
              main_runner_(*tts.main_runner_) {
            temp_ = std::sqrt(tts.cfg_.temperature);
            flow_inputs_.reserve(4);
            
            main_runner_.reinit();
            
            {
                auto _ = g_prof.time("text_conditioning");
                std::vector<Ort::Value> in;
                in.push_back(Ort::Value::CreateTensor<int64_t>(m_, const_cast<int64_t*>(tid.ptr()), tid.numel(), tid.shape.data(), tid.shape.size()));
                auto out = tts.txt_->run(in);
                
                auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t n = 1;
                for (auto d : sh) n *= d;
                temb_.assign(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n);
                tsh_.assign(sh.begin(), sh.end());
                if (tsh_.size() == 2) tsh_.insert(tsh_.begin(), 1);
            }
            
            {
                auto _ = g_prof.time("voice_conditioning_pass");
                cond_pass(v.ptr(), v.numel(), v.shape);
            }
            
            if (out_voice_snap) *out_voice_snap = main_runner_.take_snapshot();
            
            {
                auto _ = g_prof.time("text_conditioning_pass");
                cond_pass(temb_.data(), temb_.size(), tsh_);
            }
        }
        
        // Cached path: restore KV snapshot → text conditioning only
        LatentGen(PocketTTS& t, const Snapshot& voice_snap, const TensorI64& tid, int max)
            : tts(t), max_(max), m_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
              fx_(32, 0), cl_(32, std::numeric_limits<float>::quiet_NaN()),
              main_runner_(*tts.main_runner_) {
            temp_ = std::sqrt(tts.cfg_.temperature);
            flow_inputs_.reserve(4);
            
            {
                auto _ = g_prof.time("text_conditioning");
                std::vector<Ort::Value> in;
                in.push_back(Ort::Value::CreateTensor<int64_t>(m_, const_cast<int64_t*>(tid.ptr()), tid.numel(), tid.shape.data(), tid.shape.size()));
                auto out = tts.txt_->run(in);
                
                auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t n = 1;
                for (auto d : sh) n *= d;
                temb_.assign(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n);
                tsh_.assign(sh.begin(), sh.end());
                if (tsh_.size() == 2) tsh_.insert(tsh_.begin(), 1);
            }
            
            {
                auto _ = g_prof.time("voice_kv_restore");
                main_runner_.restore_snapshot(voice_snap);
            }
            
            {
                auto _ = g_prof.time("text_conditioning_pass");
                cond_pass(temb_.data(), temb_.size(), tsh_);
            }
        }
        
        bool has_next() const { return !done_ && idx_ < max_; }
        
        Tensor next() {
            if (!has_next()) throw std::runtime_error("No more latents");
            
            float eos_logit = 0;
            
            {
                auto _ = g_prof.time("frame:main_model");
                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(m_, cl_.data(), cl_.size(), curr_shape_, 3));
                inputs.push_back(Ort::Value::CreateTensor<float>(m_, nullptr, 0, empty_text_shape_, 3));
                
                auto outputs = main_runner_.run(inputs);
                
                auto csh = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t cn = 1;
                for (auto d : csh) cn *= d;
                cond_.assign(outputs[0].GetTensorData<float>(), outputs[0].GetTensorData<float>() + cn);
                csh_.assign(csh.begin(), csh.end());
                
                eos_logit = outputs[1].GetTensorData<float>()[0];
            }
            
            if (!eos_ && eos_logit > tts.cfg_.eos_threshold) {
                eos_ = true;
                eos_frame_ = idx_;
            }
            
            if (eos_) {
                if (++extra_ > tts.cfg_.eos_extra_frames) { 
                    done_ = true; 
                    return Tensor(); 
                }
            }
            
            {
                auto _ = g_prof.time("frame:rng");
                if (temp_ > 0) {
                    rng::fill_normal(fx_.data(), 32, 0, temp_);
                    float nc = tts.cfg_.noise_clamp;
                    if (nc > 0) for (auto& v : fx_) v = std::max(-nc, std::min(nc, v));
                }
                else std::fill(fx_.begin(), fx_.end(), 0.0f);
            }
            
            {
                auto _ = g_prof.time("frame:flow_steps");
                for (const auto& [s, t] : tts.st_values_) {
                    s_buf_[0] = s;
                    t_buf_[0] = t;
                    
                    flow_inputs_.clear();
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, cond_.data(), cond_.size(), csh_.data(), csh_.size()));
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, s_buf_.data(), 1, s_shape_, 2));
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, t_buf_.data(), 1, s_shape_, 2));
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, fx_.data(), 32, x_shape_, 2));
                    
                    auto fo = tts.flow_->run(flow_inputs_);
                    const float* out_data = fo[0].GetTensorData<float>();
                    
                    for (int i = 0; i < 32; ++i)
                        fx_[i] += out_data[i] * tts.dt_;
                }
            }
            
            std::copy(fx_.begin(), fx_.end(), cl_.begin());
            idx_++;
            return Tensor({fx_.begin(), fx_.end()}, {1, 1, 32});
        }
        
        int frame_idx() const { return idx_; }
        int eos_frame() const { return eos_frame_; }
    };
    
    friend class LatentGen;
    
    // ── Voice KV Cache ──────────────────────────────────────────────────────
    // Three-tier cache for voice-conditioned KV state:
    //   1. In-memory snapshot (fastest, ~1ms restore)
    //   2. On-disk .kv file (fast, ~4ms restore)
    //   3. Full recomputation (slow, hundreds of ms)
    
    using VoiceKVSnapshot = LatentGen::Snapshot;
    
    std::unique_ptr<VoiceKVSnapshot> voice_kv_snap_;
    uint64_t voice_kv_hash_ = 0;
    std::string voice_kv_path_;
    
    static uint64_t voice_hash(const Tensor& v) {
        uint64_t h = 14695981039346656037ull;
        int n = std::min(16, int(v.data.size()));
        for (int i = 0; i < n; ++i) {
            uint32_t bits;
            memcpy(&bits, &v.data[i], sizeof(bits));
            h ^= bits;
            h *= 1099511628211ull;
        }
        return h;
    }
    
    LatentGen make_gen(const Tensor& v, const TensorI64& t, int max) {
        uint64_t vh = voice_hash(v);
        
        // Tier 1: in-memory cache hit
        if (voice_kv_snap_ && voice_kv_hash_ == vh) {
            return LatentGen(*this, *voice_kv_snap_, t, max);
        }
        
        // Tier 2: disk cache hit
        if (cfg_.voice_cache && !voice_kv_path_.empty()) {
            std::string kv_path = cache::get_cache_path(cfg_.voices_dir, voice_kv_path_, "kv");
            StateBufferIO::DiskSnapshot ds;
            if (ds.load_from_disk(kv_path)) {
                if (cfg_.verbose) std::cerr << "  Loaded KV cache: " << kv_path << "\n";
                main_runner_->restore_from_disk(ds);
                voice_kv_snap_ = std::make_unique<VoiceKVSnapshot>(main_runner_->take_snapshot());
                voice_kv_hash_ = vh;
                return LatentGen(*this, *voice_kv_snap_, t, max);
            }
        }
        
        // Tier 3: full voice conditioning
        VoiceKVSnapshot snap;
        auto gen = LatentGen(*this, v, t, max, &snap);
        voice_kv_snap_ = std::make_unique<VoiceKVSnapshot>(std::move(snap));
        voice_kv_hash_ = vh;
        
        if (cfg_.voice_cache && !voice_kv_path_.empty()) {
            std::string kv_path = cache::get_cache_path(cfg_.voices_dir, voice_kv_path_, "kv");
            auto ds = main_runner_->snapshot_to_disk(*voice_kv_snap_);
            if (ds.save_to_disk(kv_path)) {
                if (cfg_.verbose) std::cerr << "  Saved KV cache: " << kv_path << "\n";
            }
        }
        
        return gen;
    }
};

// Required for C++17 ODR-use of constexpr static members
constexpr int64_t PocketTTS::LatentGen::curr_shape_[3];
constexpr int64_t PocketTTS::LatentGen::empty_text_shape_[3];
constexpr int64_t PocketTTS::LatentGen::empty_seq_shape_[3];
constexpr int64_t PocketTTS::LatentGen::s_shape_[2];
constexpr int64_t PocketTTS::LatentGen::x_shape_[2];

// ── Out-of-line method definitions ──────────────────────────────────────────

AudioData PocketTTS::generate(const std::string& text, const Tensor& voice, int max_frames) {
    auto _ = g_prof.time("generate_total");
    
    auto sentences = split_sentences(text);
    if (sentences.empty()) sentences.push_back(text);
    
    if (sentences.size() == 1) {
        std::vector<float> samples;
        samples.reserve(max_frames * 2000);
        stream(sentences[0], voice, [&](const float* s, size_t n) {
            samples.insert(samples.end(), s, s + n);
            return true;
        }, max_frames);
        return {std::move(samples), SR};
    }
    
    // Multi-sentence: generate each independently, crossfade at boundaries
    static constexpr int XFADE_SAMPLES = 240;  // 10ms at 24kHz
    
    std::vector<float> all_samples;
    
    for (size_t i = 0; i < sentences.size(); ++i) {
        if (cfg_.verbose) {
            std::cerr << "  Sentence " << (i + 1) << "/" << sentences.size() 
                      << ": \"" << sentences[i].substr(0, 60) 
                      << (sentences[i].size() > 60 ? "..." : "") << "\"\n";
        }
        
        std::vector<float> chunk_samples;
        stream(sentences[i], voice, [&](const float* s, size_t n) {
            chunk_samples.insert(chunk_samples.end(), s, s + n);
            return true;
        }, max_frames);
        
        if (chunk_samples.empty()) continue;
        
        if (i > 0 && !all_samples.empty()) {
            int xfade = std::min(XFADE_SAMPLES, std::min(int(all_samples.size()), int(chunk_samples.size())));
            size_t tail_start = all_samples.size() - xfade;
            for (int j = 0; j < xfade; ++j) {
                float t = float(j) / float(xfade);
                all_samples[tail_start + j] = all_samples[tail_start + j] * (1.0f - t) + chunk_samples[j] * t;
            }
            all_samples.insert(all_samples.end(), chunk_samples.begin() + xfade, chunk_samples.end());
        } else {
            all_samples.insert(all_samples.end(), chunk_samples.begin(), chunk_samples.end());
        }
    }
    
    return {std::move(all_samples), SR};
}

void PocketTTS::stream(const std::string& text, const std::string& voice, StreamCallback cb, int max_frames) {
    stream(text, get_voice(voice), cb, max_frames);
}

void PocketTTS::stream(const std::string& text, const Tensor& voice, StreamCallback cb, int max_frames) {
    auto sentences = split_sentences(text);
    if (sentences.empty()) sentences.push_back(text);
    
    for (size_t si = 0; si < sentences.size(); ++si) {
        auto gen = make_gen(voice, tokenize(sentences[si]), max_frames);
        StatefulRunner dec_runner(*dec_);
        
        // Pipelined: generator thread produces latent frames into a queue,
        // decoder (main thread) consumes them in chunks. The two ONNX sessions
        // (flow_lm_main and mimi_decoder) run on separate threads simultaneously.
        
        std::mutex mtx;
        std::condition_variable cv;
        std::deque<Tensor> queue;
        bool gen_done = false;
        bool aborted = false;
        
        std::thread gen_thread([&]() {
            while (gen.has_next()) {
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (aborted) return;
                }
                auto f = gen.next();
                if (f.numel() == 0) break;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    queue.push_back(std::move(f));
                }
                cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                gen_done = true;
            }
            cv.notify_one();
        });
        
        bool first = true;
        
        while (true) {
            int want = first ? cfg_.first_chunk_frames : cfg_.max_chunk_frames;
            
            std::vector<Tensor> batch;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]{ return (int)queue.size() >= want || gen_done; });
                
                int take = gen_done ? (int)queue.size() : std::min((int)queue.size(), want);
                for (int i = 0; i < take; ++i) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }
            
            if (batch.empty()) break;
            
            auto lat = Tensor::concat(batch, 1);
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<float>(dec_runner.mem(), lat.ptr(), lat.numel(),
                                                              lat.shape.data(), lat.shape.size()));
            
            auto outputs = dec_runner.run(inputs);
            auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t n = 1;
            for (auto d : shape) n *= d;
            
            if (!cb(outputs[0].GetTensorData<float>(), n)) {
                std::lock_guard<std::mutex> lock(mtx);
                aborted = true;
                break;
            }
            first = false;
        }
        
        gen_thread.join();
        if (aborted) return;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HTTP Server
// ════════════════════════════════════════════════════════════════════════════

static std::atomic<bool> g_server_running{true};
static int g_server_fd = -1;

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    
    static HttpRequest parse(int client_fd) {
        HttpRequest req;
        std::string data;
        char buf[4096];
        
        while (true) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            data.append(buf, n);
            
            size_t header_end = data.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                size_t cl_pos = data.find("Content-Length:");
                if (cl_pos == std::string::npos) cl_pos = data.find("content-length:");
                
                if (cl_pos != std::string::npos) {
                    size_t cl_end = data.find("\r\n", cl_pos);
                    int content_length = std::stoi(data.substr(cl_pos + 15, cl_end - cl_pos - 15));
                    size_t body_start = header_end + 4;
                    
                    while (data.size() < body_start + content_length) {
                        n = recv(client_fd, buf, sizeof(buf), 0);
                        if (n <= 0) break;
                        data.append(buf, n);
                    }
                }
                break;
            }
        }
        
        size_t line_end = data.find("\r\n");
        if (line_end != std::string::npos) {
            std::string line = data.substr(0, line_end);
            size_t sp1 = line.find(' ');
            size_t sp2 = line.find(' ', sp1 + 1);
            if (sp1 != std::string::npos && sp2 != std::string::npos) {
                req.method = line.substr(0, sp1);
                req.path = line.substr(sp1 + 1, sp2 - sp1 - 1);
            }
        }
        
        size_t body_start = data.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            req.body = data.substr(body_start + 4);
        }
        
        return req;
    }
};

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    
    pos = json.find('"', pos);
    if (pos == std::string::npos) return "";
    
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    
    return json.substr(pos + 1, end - pos - 1);
}

class TTSServer {
    PocketTTS& tts_;
    int port_;
    int server_fd_ = -1;
    std::mutex tts_mutex_;
    
public:
    TTSServer(PocketTTS& tts, int port) : tts_(tts), port_(port) {}
    
    ~TTSServer() {
        if (server_fd_ >= 0 && server_fd_ == g_server_fd) {
            close(server_fd_);
            g_server_fd = -1;
        }
        server_fd_ = -1;
    }
    
    bool start() {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            std::cerr << "Failed to create socket\n";
            return false;
        }
        g_server_fd = server_fd_;
        
        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);
        
        if (bind(server_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Failed to bind to port " << port_ << "\n";
            return false;
        }
        
        if (listen(server_fd_, 5) < 0) {
            std::cerr << "Failed to listen\n";
            return false;
        }
        
        std::cout << "TTS Server listening on http://localhost:" << port_ << "\n";
        std::cout << "Endpoints:\n";
        std::cout << "  POST /tts          - Generate speech (JSON: {\"text\": \"...\", \"voice\": \"...\"})\n";
        std::cout << "  GET  /health       - Health check\n";
        std::cout << "Press Ctrl+C to stop\n\n";
        
        return true;
    }
    
    void run() {
        while (g_server_running) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) break;
            
            struct timeval tv;
            tv.tv_sec = 30;
            tv.tv_usec = 0;
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            
            handle_request(client_fd);
            close(client_fd);
        }
    }
    
private:
    void send_response(int fd, int status, const std::string& content_type, const std::string& body) {
        std::string status_text = (status == 200) ? "OK" : (status == 404) ? "Not Found" : "Bad Request";
        std::ostringstream resp;
        resp << "HTTP/1.1 " << status << " " << status_text << "\r\n";
        resp << "Content-Type: " << content_type << "\r\n";
        resp << "Content-Length: " << body.size() << "\r\n";
        resp << "Access-Control-Allow-Origin: *\r\n";
        resp << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        resp << "Access-Control-Allow-Headers: Content-Type\r\n";
        resp << "\r\n";
        resp << body;
        
        std::string data = resp.str();
        send(fd, data.c_str(), data.size(), 0);
    }
    
    void send_chunked_header(int fd, const std::string& content_type) {
        std::ostringstream resp;
        resp << "HTTP/1.1 200 OK\r\n";
        resp << "Content-Type: " << content_type << "\r\n";
        resp << "Transfer-Encoding: chunked\r\n";
        resp << "Access-Control-Allow-Origin: *\r\n";
        resp << "\r\n";
        
        std::string data = resp.str();
        send(fd, data.c_str(), data.size(), 0);
    }
    
    void send_chunk(int fd, const void* data, size_t len) {
        char size_buf[32];
        snprintf(size_buf, sizeof(size_buf), "%zx\r\n", len);
        send(fd, size_buf, strlen(size_buf), 0);
        send(fd, data, len, 0);
        send(fd, "\r\n", 2, 0);
    }
    
    void send_final_chunk(int fd) {
        send(fd, "0\r\n\r\n", 5, 0);
    }
    
    void handle_request(int client_fd) {
        auto req = HttpRequest::parse(client_fd);
        
        char client_ip[INET_ADDRSTRLEN];
        sockaddr_in addr;
        socklen_t len = sizeof(addr);
        getpeername(client_fd, (sockaddr*)&addr, &len);
        inet_ntop(AF_INET, &addr.sin_addr, client_ip, sizeof(client_ip));
        std::cout << client_ip << " " << req.method << " " << req.path << "\n";
        
        if (req.method == "OPTIONS") {
            send_response(client_fd, 200, "text/plain", "");
            return;
        }
        
        if (req.method == "GET" && req.path == "/health") {
            send_response(client_fd, 200, "application/json", "{\"status\":\"ok\"}");
        }
        else if (req.method == "POST" && req.path == "/tts") {
            std::string text = json_get_string(req.body, "text");
            std::string voice = json_get_string(req.body, "voice");
            
            if (text.empty() || voice.empty()) {
                send_response(client_fd, 400, "application/json", "{\"error\":\"Missing text or voice\"}");
                return;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            std::cout << "  Generating: \"" << text << "\" with voice '" << voice << "'\n";
            
            try {
                send_chunked_header(client_fd, "audio/pcm;rate=24000;encoding=float;bits=32");
                
                bool first_chunk = true;
                size_t total_samples = 0;
                
                {
                    std::lock_guard<std::mutex> lock(tts_mutex_);
                    tts_.stream(text, voice, [&](const float* samples, size_t n) {
                        if (first_chunk) {
                            auto now = std::chrono::high_resolution_clock::now();
                            double latency = std::chrono::duration<double, std::milli>(now - start).count();
                            std::cout << "  First chunk latency: " << std::fixed << std::setprecision(0) << latency << "ms\n";
                            first_chunk = false;
                        }
                        send_chunk(client_fd, samples, n * sizeof(float));
                        total_samples += n;
                        return true;
                    });
                }
                
                send_final_chunk(client_fd);
                
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(end - start).count();
                double duration = double(total_samples) / PocketTTS::SR;
                std::cout << "  Done: " << std::fixed << std::setprecision(2) << duration << "s audio in " << elapsed << "s (RTFx: " << duration/elapsed << "x)\n";
            } catch (const std::exception& e) {
                send_response(client_fd, 400, "application/json", "{\"error\":\"" + std::string(e.what()) + "\"}");
            }
        }
        else {
            send_response(client_fd, 404, "application/json", "{\"error\":\"Not found\"}");
        }
    }
};

} // namespace pocket_tts

// ════════════════════════════════════════════════════════════════════════════
// C API (FFI)
// ════════════════════════════════════════════════════════════════════════════

extern "C" {

void* ptt_create(const char* models_dir, const char* voices_dir,
                 const char* tokenizer_path, const char* precision,
                 float temperature, int lsd_steps, int num_threads) {
    try {
        pocket_tts::Config cfg;
        if (models_dir) cfg.models_dir = models_dir;
        if (voices_dir) cfg.voices_dir = voices_dir;
        if (tokenizer_path) cfg.tokenizer_path = tokenizer_path;
        if (precision) cfg.precision = precision;
        cfg.temperature = temperature;
        cfg.lsd_steps = lsd_steps;
        cfg.num_threads = num_threads;
        return new pocket_tts::PocketTTS(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[pocket-tts] init error: " << e.what() << "\n";
        return nullptr;
    }
}

double ptt_warmup(void* handle) {
    if (!handle) return -1;
    try {
        return static_cast<pocket_tts::PocketTTS*>(handle)->warmup();
    } catch (const std::exception& e) {
        std::cerr << "[pocket-tts] warmup error: " << e.what() << "\n";
        return -1;
    }
}

int ptt_synthesize(void* handle, const char* text, const char* voice,
                   float** out_samples, int* out_len, int* out_sample_rate) {
    if (!handle || !text || !voice || !out_samples || !out_len || !out_sample_rate) return -1;
    try {
        auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
        auto audio = tts->generate(text, voice);
        *out_sample_rate = audio.sample_rate;
        *out_len = static_cast<int>(audio.samples.size());
        *out_samples = static_cast<float*>(malloc(*out_len * sizeof(float)));
        if (!*out_samples) return -2;
        std::memcpy(*out_samples, audio.samples.data(), *out_len * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[pocket-tts] synthesize error: " << e.what() << "\n";
        return -1;
    }
}

void ptt_free_audio(float* samples) {
    free(samples);
}

void ptt_destroy(void* handle) {
    delete static_cast<pocket_tts::PocketTTS*>(handle);
}

// ── Streaming API ───────────────────────────────────────────────────────────
// ptt_stream_start  — kicks off generation on a background thread
// ptt_stream_read   — blocks until next decoded chunk (or stream end)
// ptt_stream_end    — join thread, free remaining chunks

struct ptt_stream_ctx {
    std::thread thread;
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::pair<float*, size_t>> chunks;
    bool done = false;
    bool aborted = false;
};

void* ptt_stream_start(void* handle, const char* text, const char* voice) {
    if (!handle || !text || !voice) return nullptr;
    auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
    auto* ctx = new ptt_stream_ctx();

    ctx->thread = std::thread([tts, t = std::string(text), v = std::string(voice), ctx]() {
        try {
            tts->stream(t, v, [ctx](const float* samples, size_t n) -> bool {
                float* copy = static_cast<float*>(malloc(n * sizeof(float)));
                if (!copy) return false;
                std::memcpy(copy, samples, n * sizeof(float));
                {
                    std::lock_guard<std::mutex> lock(ctx->mtx);
                    if (ctx->aborted) { free(copy); return false; }
                    ctx->chunks.push_back({copy, n});
                }
                ctx->cv.notify_one();
                return true;
            });
        } catch (const std::exception& e) {
            std::cerr << "[pocket-tts] stream error: " << e.what() << "\n";
        }
        {
            std::lock_guard<std::mutex> lock(ctx->mtx);
            ctx->done = true;
        }
        ctx->cv.notify_one();
    });

    return ctx;
}

int ptt_stream_read(void* stream_ctx, float** out_samples, int* out_len) {
    if (!stream_ctx || !out_samples || !out_len) return -1;
    auto* ctx = static_cast<ptt_stream_ctx*>(stream_ctx);

    std::unique_lock<std::mutex> lock(ctx->mtx);
    ctx->cv.wait(lock, [ctx]{ return !ctx->chunks.empty() || ctx->done; });

    if (!ctx->chunks.empty()) {
        auto [ptr, len] = ctx->chunks.front();
        ctx->chunks.pop_front();
        *out_samples = ptr;
        *out_len = static_cast<int>(len);
        return 1;
    }
    return 0;
}

void ptt_stream_end(void* stream_ctx) {
    if (!stream_ctx) return;
    auto* ctx = static_cast<ptt_stream_ctx*>(stream_ctx);
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->aborted = true;
    }
    ctx->cv.notify_all();
    if (ctx->thread.joinable()) ctx->thread.join();
    for (auto& [ptr, len] : ctx->chunks) free(ptr);
    delete ctx;
}

} // extern "C"

// ════════════════════════════════════════════════════════════════════════════
// CLI + HTTP Server Entry Point
// ════════════════════════════════════════════════════════════════════════════

#ifndef PTT_SHARED_LIB

static void signal_handler(int sig) {
    (void)sig;
    pocket_tts::g_server_running = false;
    if (pocket_tts::g_server_fd >= 0) {
        close(pocket_tts::g_server_fd);
        pocket_tts::g_server_fd = -1;
    }
    std::cout << "\nShutting down...\n";
}

int main(int argc, char* argv[]) {
    pocket_tts::Config cfg;
    bool stdout_output = false;
    bool server_mode = false;
    int server_port = 8080;
    std::string text, voice, output;
    int pos = 0;
    
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> char* {
            if (++i >= argc) { std::cerr << "Missing value for " << a << "\n"; exit(1); }
            return argv[i];
        };
        if (a == "-h" || a == "--help") {
            std::cerr << "Usage: " << argv[0] << " [OPTIONS] TEXT VOICE [OUTPUT]\n"
                "       " << argv[0] << " --server [OPTIONS]\n"
                "\nOptions:\n"
                "  --precision <int8|fp32>  Model precision (default: int8)\n"
                "  --temperature <float>    Sampling temperature (default: 0.7)\n"
                "  --lsd-steps <int>        Flow matching steps (default: 1)\n"
                "  --threads <int>          Total thread budget (default: 0 = half cores)\n"
                "  --models-dir <path>      ONNX models directory (default: models)\n"
                "  --voices-dir <path>      Voice samples directory (default: voices)\n"
                "  --tokenizer <path>       Tokenizer path (default: models/tokenizer.model)\n"
                "  --eos-threshold <float>  EOS detection threshold (default: -4.0)\n"
                "  --noise-clamp <float>    Noise clamp value (default: 3.0, 0=disabled)\n"
                "  --eos-extra <int>        Extra frames after EOS (default: 10)\n"
                "  --first-chunk <int>      Frames in first decode chunk (default: 1)\n"
                "  --max-chunk <int>        Max frames per decode chunk (default: 15)\n"
                "  --no-cache               Disable all disk caching (.emb and .kv files)\n"
                "\nOutput:\n"
                "  --stdout                 Output raw f32le PCM to stdout (for piping)\n"
                "  --verbose                Enable verbose output\n"
                "  --profile                Show profiling report with first-chunk latency\n"
                "\nServer mode:\n"
                "  --server                 Start HTTP server (models prewarmed on startup)\n"
                "  --port <port>            Server port (default: 8080)\n";
            return 0;
        }
        else if (a == "--precision") cfg.precision = next();
        else if (a == "--temperature") cfg.temperature = std::stof(next());
        else if (a == "--lsd-steps") cfg.lsd_steps = std::stoi(next());
        else if (a == "--threads") cfg.num_threads = std::stoi(next());
        else if (a == "--models-dir") cfg.models_dir = next();
        else if (a == "--voices-dir") cfg.voices_dir = next();
        else if (a == "--tokenizer") cfg.tokenizer_path = next();
        else if (a == "--eos-threshold") cfg.eos_threshold = std::stof(next());
        else if (a == "--noise-clamp") cfg.noise_clamp = std::stof(next());
        else if (a == "--eos-extra") cfg.eos_extra_frames = std::stoi(next());
        else if (a == "--first-chunk") cfg.first_chunk_frames = std::stoi(next());
        else if (a == "--max-chunk") cfg.max_chunk_frames = std::stoi(next());
        else if (a == "--no-cache") cfg.voice_cache = false;
        else if (a == "--stdout") stdout_output = true;
        else if (a == "--verbose") cfg.verbose = true;
        else if (a == "--profile") pocket_tts::g_prof.enabled = true;
        else if (a == "--server") server_mode = true;
        else if (a == "--port") server_port = std::stoi(next());
        else if (a[0] == '-') { std::cerr << "Unknown: " << a << "\n"; return 1; }
        else { if (pos == 0) text = a; else if (pos == 1) voice = a; else if (pos == 2) output = a; pos++; }
    }
    
    if (!server_mode) {
        if (pos < 2) { std::cerr << "Need: TEXT VOICE [OUTPUT]\n"; return 1; }
        if (pos < 3 && !stdout_output) { std::cerr << "Need OUTPUT file (or use --stdout)\n"; return 1; }
        
        if (stdout_output) {
            cfg.verbose = false;
            pocket_tts::g_prof.enabled = false;
        }
    }
    
    try {
        int threads = cfg.num_threads ? cfg.num_threads : std::max(2, int(std::thread::hardware_concurrency()) / 2);
        
        if (!stdout_output) {
            std::cerr << "Loading (precision=" << cfg.precision << ", threads=" << threads << ")...\n";
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        pocket_tts::PocketTTS tts(cfg);
        auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); };
        
        if (!stdout_output) {
            std::cerr << "  Loaded in " << std::fixed << std::setprecision(2) << elapsed() << "s\n";
        }
        
        if (server_mode) {
            double warmup_ms = tts.warmup();
            std::cerr << "  Warmup in " << std::fixed << std::setprecision(0) << warmup_ms << "ms\n";
            
            signal(SIGINT, signal_handler);
            signal(SIGTERM, signal_handler);
            
            pocket_tts::TTSServer server(tts, server_port);
            if (!server.start()) return 1;
            server.run();
        }
        else {
            tts.reset_profiling();
            
            if (!stdout_output) {
                std::cerr << "Generating: \"" << text << "\" with " << voice << "\n";
            }
            t0 = std::chrono::high_resolution_clock::now();
            
            pocket_tts::AudioData audio;
            double first_chunk_latency = 0;
            
            if (stdout_output) {
                size_t total_samples = 0;
                bool first = true;
                tts.stream(text, voice, [&](const float* s, size_t n) {
                    if (first) {
                        first_chunk_latency = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                        first = false;
                    }
                    fwrite(s, sizeof(float), n, stdout);
                    fflush(stdout);
                    total_samples += n;
                    return true;
                });
                audio.sample_rate = pocket_tts::PocketTTS::SR;
                audio.samples.resize(total_samples);
            } else {
                if (pocket_tts::g_prof.enabled) {
                    std::vector<float> samples;
                    bool first = true;
                    tts.stream(text, voice, [&](const float* s, size_t n) {
                        if (first) {
                            first_chunk_latency = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                            first = false;
                        }
                        samples.insert(samples.end(), s, s + n);
                        return true;
                    });
                    audio = {std::move(samples), pocket_tts::PocketTTS::SR};
                } else {
                    audio = tts.generate(text, voice);
                }
            }
            
            double gen_time = elapsed();
            double duration = audio.duration_sec();
            
            if (!stdout_output) {
                std::cerr << "  " << std::fixed << std::setprecision(2) 
                          << duration << "s audio in " << gen_time << "s (RTFx: " << duration / gen_time << "x)\n";
                if (pocket_tts::g_prof.enabled) {
                    std::cerr << "  First chunk latency: " << std::fixed << std::setprecision(0) 
                              << first_chunk_latency << "ms\n";
                }
                pocket_tts::PocketTTS::save_audio(audio, output);
                std::cerr << "  Saved: " << output << "\n";
            } else {
                std::cerr << "  " << std::fixed << std::setprecision(2)
                          << duration << "s audio in " << gen_time << "s (RTFx: " << duration / gen_time << "x)\n";
                if (pocket_tts::g_prof.enabled) {
                    std::cerr << "  First chunk latency: " << std::fixed << std::setprecision(0) 
                              << first_chunk_latency << "ms\n";
                }
            }
            
            if (pocket_tts::g_prof.enabled) {
                tts.print_profiling_report();
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

#endif // PTT_SHARED_LIB
