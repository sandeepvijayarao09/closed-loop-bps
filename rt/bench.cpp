// Latency benchmark + correctness check for the streaming decoders.
//
//   bench <model.bin> <ref.bin> [--steps N] [--paced-seconds S] [--json out.json] [--label NAME]
//
// 1. Correctness: replays the reference spike sequence exported from Python and
//    compares every output against the Python (float64) decoder.
// 2. Throughput latency: times step() individually for N bins (after warm-up)
//    and reports p50 / p99 / p99.9 / max in microseconds.
// 3. Paced loop: runs the decoder on a real 20 ms tick for S
//    seconds (sleep-based, then spin-based) and reports wake-up lateness,
//    compute time and deadline misses.
//    macOS is not a real-time OS; this measures what a user-space loop gets.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "decoders.hpp"

using clk = std::chrono::steady_clock;

struct Ref {
    uint32_t T = 0, U = 0;
    std::vector<float> X, Y;
};

static Ref load_ref(const std::string& path) {
    cbps::detail::Reader r(path);
    Ref ref;
    ref.T = r.u32();
    ref.U = r.u32();
    ref.X = r.floats(size_t(ref.T) * ref.U);
    ref.Y = r.floats(size_t(ref.T) * 2);
    return ref;
}

static double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size() - 1, size_t(std::ceil(q * v.size())) - 1);
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.bin ref.bin [--steps N] [--paced-seconds S] [--json out]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1], ref_path = argv[2], json_path, label;
    size_t steps = 100000;
    double paced_s = 0.0;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::stoul(argv[++i]);
        else if (!std::strcmp(argv[i], "--paced-seconds") && i + 1 < argc) paced_s = std::stod(argv[++i]);
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!std::strcmp(argv[i], "--label") && i + 1 < argc) label = argv[++i];
    }
    auto dec = cbps::load(model_path);
    Ref ref = load_ref(ref_path);
    if (ref.U != dec->n_units) {
        std::fprintf(stderr, "unit mismatch: model %u ref %u\n", dec->n_units, ref.U);
        return 1;
    }

    // 1. correctness vs Python
    double max_abs = 0.0, max_ref = 0.0;
    float out[2];
    dec->reset();
    for (uint32_t t = 0; t < ref.T; ++t) {
        dec->step(&ref.X[size_t(t) * ref.U], out);
        for (int d = 0; d < 2; ++d) {
            max_abs = std::max(max_abs, double(std::fabs(out[d] - ref.Y[2 * t + d])));
            max_ref = std::max(max_ref, double(std::fabs(ref.Y[2 * t + d])));
        }
    }
    double rel = max_abs / std::max(max_ref, 1e-9);
    bool ok = rel < 1e-4;

    // 2. per-step latency (inputs cycle through the real reference spikes)
    dec->reset();
    for (size_t i = 0; i < 2000; ++i) dec->step(&ref.X[size_t(i % ref.T) * ref.U], out);  // warm-up
    std::vector<double> us(steps);
    volatile float sink = 0.f;
    for (size_t i = 0; i < steps; ++i) {
        const float* x = &ref.X[size_t(i % ref.T) * ref.U];
        auto a = clk::now();
        dec->step(x, out);
        auto b = clk::now();
        sink = sink + out[0];
        us[i] = std::chrono::duration<double, std::micro>(b - a).count();
    }

    // 3. paced 20 ms loop, two ways of waiting for the tick:
    //    sleep: std::this_thread::sleep_until (what an ordinary user-space loop gets;
    //           macOS coalesces timers, so wake-ups are milliseconds late)
    //    spin:  busy-wait on the clock (what a pinned real-time thread would do; burns a core)
    struct Paced { std::vector<double> late, total; size_t misses = 0; };
    auto paced = [&](bool spin) {
        Paced p;
        size_t ticks = size_t(paced_s / 0.02);
        auto next = clk::now() + std::chrono::milliseconds(20);
        for (size_t i = 0; i < ticks; ++i) {
            if (spin) {
                while (clk::now() < next) {
                }
            } else {
                std::this_thread::sleep_until(next);
            }
            auto woke = clk::now();
            dec->step(&ref.X[size_t(i % ref.T) * ref.U], out);
            auto done = clk::now();
            p.late.push_back(std::chrono::duration<double, std::micro>(woke - next).count());
            p.total.push_back(std::chrono::duration<double, std::micro>(done - next).count());
            if (p.total.back() > 20000.0) ++p.misses;
            next += std::chrono::milliseconds(20);
        }
        return p;
    };
    Paced ps, pp;
    if (paced_s > 0) {
        ps = paced(false);
        pp = paced(true);
    }

    if (label.empty()) label = dec->name();
    double p50 = pct(us, 0.50), p99 = pct(us, 0.99), p999 = pct(us, 0.999),
           mx = *std::max_element(us.begin(), us.end());
    std::printf("%-10s params=%-8zu check=%s (max rel err %.2e)  step us: p50 %.2f  p99 %.2f  p99.9 %.2f  max %.1f\n",
                label.c_str(), dec->n_params(), ok ? "PASS" : "FAIL", rel, p50, p99, p999, mx);
    for (auto* p : {&ps, &pp})
        if (!p->late.empty())
            std::printf("           paced 20 ms x %zu (%s): lateness p50 %.0f us p99 %.0f us, wake+compute p99 %.0f us, "
                        "deadline misses %zu\n",
                        p->late.size(), p == &ps ? "sleep" : "spin", pct(p->late, 0.5), pct(p->late, 0.99),
                        pct(p->total, 0.99), p->misses);
    if (!json_path.empty()) {
        FILE* f = std::fopen(json_path.c_str(), "w");
        std::fprintf(f,
                     "{\"name\": \"%s\", \"n_params\": %zu, \"check_pass\": %s, \"max_rel_err\": %.3e, "
                     "\"steps\": %zu, \"p50_us\": %.3f, \"p99_us\": %.3f, \"p999_us\": %.3f, \"max_us\": %.3f",
                     label.c_str(), dec->n_params(), ok ? "true" : "false", rel, steps, p50, p99, p999, mx);
        for (auto* p : {&ps, &pp})
            if (!p->late.empty()) {
                const char* m = p == &ps ? "sleep" : "spin";
                std::fprintf(f,
                             ", \"paced_%s\": {\"ticks\": %zu, \"lateness_p50_us\": %.1f, \"lateness_p99_us\": %.1f, "
                             "\"total_p99_us\": %.1f, \"total_max_us\": %.1f, \"deadline_misses\": %zu}",
                             m, p->late.size(), pct(p->late, 0.5), pct(p->late, 0.99), pct(p->total, 0.99),
                             *std::max_element(p->total.begin(), p->total.end()), p->misses);
            }
        std::fprintf(f, "}\n");
        std::fclose(f);
    }
    return ok ? 0 : 1;
}
