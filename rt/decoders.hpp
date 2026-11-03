// Real-time streaming decoders: one 20 ms bin of spike counts in, 2-D velocity out.
//
// Dependency-free C++17, float32, all buffers allocated at load time: step()
// never allocates, locks or does I/O, so its latency is just arithmetic.
// Weight files are written by cbps/export.py via scripts/05_export_bench.py
// (format documented in cbps/export.py).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cbps {

enum Kind : uint32_t { KF = 1, GRU = 2, WIENER = 3 };

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual void reset() = 0;
    // counts: n_units floats; out: 2 floats (mm/s)
    virtual void step(const float* counts, float* out) = 0;
    virtual size_t n_params() const = 0;
    virtual std::string name() const = 0;
    uint32_t n_units = 0;
};

namespace detail {
struct Reader {
    FILE* f;
    explicit Reader(const std::string& path) : f(std::fopen(path.c_str(), "rb")) {
        if (!f) throw std::runtime_error("cannot open " + path);
    }
    ~Reader() { if (f) std::fclose(f); }
    uint32_t u32() {
        uint32_t v;
        if (std::fread(&v, 4, 1, f) != 1) throw std::runtime_error("truncated file");
        return v;
    }
    void expect_eof() {
        if (std::fgetc(f) != EOF) throw std::runtime_error("trailing bytes after weights (format mismatch)");
    }
    std::vector<float> floats(size_t n) {
        std::vector<float> v(n);
        if (n && std::fread(v.data(), 4, n, f) != n) throw std::runtime_error("truncated file");
        return v;
    }
};

// y[r] = sum_c A[r*cols + c] * x[c]   (row-major mat-vec; the hot loop)
inline void matvec(const float* A, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* a = A + r * cols;
        float acc = 0.f;
        for (size_t c = 0; c < cols; ++c) acc += a[c] * x[c];
        y[r] = acc;
    }
}
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
}  // namespace detail

// Steady-state Kalman filter, d-dim state (2 = velocity KF / ReFIT-KF, more =
// lag-augmented KF-lag): x = M1 x + M2 (y - mu); output x[out0], x[out1].
class KalmanDecoder : public Decoder {
public:
    uint32_t d = 2, out0 = 0, out1 = 1;
    std::vector<float> mu, M1, M2, yc, x, xn, k;
    void reset() override { std::fill(x.begin(), x.end(), 0.f); }
    void step(const float* counts, float* out) override {
        for (uint32_t i = 0; i < n_units; ++i) yc[i] = counts[i] - mu[i];
        detail::matvec(M2.data(), yc.data(), k.data(), d, n_units);
        detail::matvec(M1.data(), x.data(), xn.data(), d, d);
        for (uint32_t i = 0; i < d; ++i) x[i] = xn[i] + k[i];
        out[0] = x[out0];
        out[1] = x[out1];
    }
    size_t n_params() const override { return mu.size() + M1.size() + M2.size(); }
    std::string name() const override { return d == 2 ? "KF" : "KF-lag"; }
};

// Wiener filter over the last n_hist bins: v = W [y_t - mu, ..., y_{t-H+1} - mu] + b
// History lives in a double buffer: each bin is written twice, n_hist slots
// apart, so the newest-first window is always one contiguous run of
// n_hist * n_units floats and the inner loop is a plain dot product.
class WienerDecoder : public Decoder {
public:
    uint32_t n_hist = 0;
    std::vector<float> mu, W, b, ring;  // W: [2, n_hist * n_units], lag-major like the Python design matrix
    uint32_t head = 0;                   // slot of the newest bin, in [0, n_hist)
    void reset() override { std::fill(ring.begin(), ring.end(), 0.f); head = 0; }
    void step(const float* counts, float* out) override {
        head = (head + n_hist - 1) % n_hist;
        float* s0 = ring.data() + size_t(head) * n_units;
        float* s1 = s0 + size_t(n_hist) * n_units;
        for (uint32_t i = 0; i < n_units; ++i) s0[i] = s1[i] = counts[i] - mu[i];
        const size_t len = size_t(n_hist) * n_units;
        for (int dd = 0; dd < 2; ++dd) {
            const float* w = W.data() + size_t(dd) * len;
            float acc = b[dd];
            for (size_t j = 0; j < len; ++j) acc += w[j] * s0[j];
            out[dd] = acc;
        }
    }
    size_t n_params() const override { return mu.size() + W.size() + b.size(); }
    std::string name() const override { return "Wiener"; }
};

// Single-layer GRU, PyTorch gate order (r, z, n), linear readout.
class GRUDecoder : public Decoder {
public:
    uint32_t H = 0;
    std::vector<float> mu, inv_sd, W_ih, W_hh, b_ih, b_hh, W_o, b_o, v_sd;
    std::vector<float> x, h, gi, gh;
    void reset() override { std::fill(h.begin(), h.end(), 0.f); }
    void step(const float* counts, float* out) override {
        for (uint32_t i = 0; i < n_units; ++i) x[i] = (counts[i] - mu[i]) * inv_sd[i];
        detail::matvec(W_ih.data(), x.data(), gi.data(), 3 * H, n_units);
        detail::matvec(W_hh.data(), h.data(), gh.data(), 3 * H, H);
        for (uint32_t j = 0; j < H; ++j) {
            float r = detail::sigmoid(gi[j] + b_ih[j] + gh[j] + b_hh[j]);
            float z = detail::sigmoid(gi[H + j] + b_ih[H + j] + gh[H + j] + b_hh[H + j]);
            float n = std::tanh(gi[2 * H + j] + b_ih[2 * H + j] + r * (gh[2 * H + j] + b_hh[2 * H + j]));
            h[j] = (1.f - z) * n + z * h[j];
        }
        float o[2];
        detail::matvec(W_o.data(), h.data(), o, 2, H);
        out[0] = (o[0] + b_o[0]) * v_sd[0];
        out[1] = (o[1] + b_o[1]) * v_sd[1];
    }
    size_t n_params() const override {
        return mu.size() + inv_sd.size() + W_ih.size() + W_hh.size() + b_ih.size() + b_hh.size() + W_o.size() +
               b_o.size();
    }
    std::string name() const override { return "GRU-" + std::to_string(H); }
};

// File: "CBPS" | u32 version=1 | u32 kind | u32 n_units | u32 param | float32 arrays...
// Truncated files and files with bytes left over both fail to load.
inline std::unique_ptr<Decoder> load(const std::string& path) {
    detail::Reader r(path);
    char magic[4];
    if (std::fread(magic, 1, 4, r.f) != 4 || std::memcmp(magic, "CBPS", 4) != 0)
        throw std::runtime_error("bad magic in " + path);
    if (r.u32() != 1) throw std::runtime_error("unsupported version");
    uint32_t kind = r.u32(), U = r.u32(), P = r.u32();
    if (kind == KF) {
        auto d = std::make_unique<KalmanDecoder>();
        d->n_units = U;
        d->d = P ? P : 2;
        d->mu = r.floats(U);
        d->M1 = r.floats(size_t(d->d) * d->d);
        d->M2 = r.floats(size_t(d->d) * U);
        auto o = r.floats(2);
        d->out0 = uint32_t(o[0]);
        d->out1 = uint32_t(o[1]);
        if (d->out0 >= d->d || d->out1 >= d->d) throw std::runtime_error("bad output index");
        d->yc.assign(U, 0.f);
        d->x.assign(d->d, 0.f);
        d->xn.assign(d->d, 0.f);
        d->k.assign(d->d, 0.f);
        r.expect_eof();
        return d;
    }
    if (kind == WIENER) {
        if (P == 0) throw std::runtime_error("n_hist must be > 0");
        auto d = std::make_unique<WienerDecoder>();
        d->n_units = U;
        d->n_hist = P;
        d->mu = r.floats(U);
        d->W = r.floats(2 * size_t(P) * U);
        d->b = r.floats(2);
        d->ring.assign(2 * size_t(P) * U, 0.f);
        r.expect_eof();
        return d;
    }
    if (kind == GRU) {
        auto d = std::make_unique<GRUDecoder>();
        d->n_units = U;
        d->H = P;
        d->mu = r.floats(U);
        d->inv_sd = r.floats(U);
        d->W_ih = r.floats(3 * size_t(P) * U);
        d->W_hh = r.floats(3 * size_t(P) * P);
        d->b_ih = r.floats(3 * P);
        d->b_hh = r.floats(3 * P);
        d->W_o = r.floats(2 * P);
        d->b_o = r.floats(2);
        d->v_sd = r.floats(2);
        d->x.assign(U, 0.f);
        d->h.assign(P, 0.f);
        d->gi.assign(3 * P, 0.f);
        d->gh.assign(3 * P, 0.f);
        r.expect_eof();
        return d;
    }
    throw std::runtime_error("unknown decoder kind");
}

}  // namespace cbps
