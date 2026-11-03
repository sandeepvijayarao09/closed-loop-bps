// Latency benchmark + correctness check for the streaming decoders.
//
//   bench <model.bin> <ref.bin> [--steps N] [--block B] [--paced-seconds S]
//         [--rt] [--json out.json] [--label NAME]
//
// 1. Correctness: replays the reference spike sequence exported from Python and
//    compares every output against the Python (float64) decoder.
// 2. Compute latency, two ways:
//    - batched: time blocks of B consecutive step() calls and divide by B.
//      This is the number to trust for fast decoders: a single steady_clock
//      read costs/quantises at tens of ns, so per-call timing of a 100 ns
//      KF step mostly measures the clock.
//    - per call: time each step() on its own, for the tail (p99.9 / max),
//      which is what a deadline cares about. Clock resolution is reported.
// 3. Paced loop: the decoder on a real 20 ms tick for S seconds, waiting for
//    each tick by sleeping (sleep_until) and then by spinning on the clock,
//    reporting wake-up lateness, wake+compute time and deadline misses.
//    By default the sleep is std::this_thread::sleep_until, i.e. what an
//    ordinary process gets; on macOS that is subject to timer coalescing
//    (~4 ms late here; USER_INTERACTIVE QoS did not change it). --rt (macOS)
//    puts the thread under THREAD_TIME_CONSTRAINT_POLICY (period 20 ms,
//    computation 1 ms, constraint 2 ms) and waits with mach_wait_until, the
//    platform's real-time mechanism for audio-style periodic threads.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#endif

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
    r.expect_eof();
    return ref;
}

static double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size() - 1, size_t(std::ceil(q * v.size())) - 1);
    return v[i];
}

static double us(clk::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

struct Paced {
    std::vector<double> late, total;
    size_t misses = 0;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s model.bin ref.bin [--steps N] [--block B] [--paced-seconds S] "
                     "[--rt] [--json out] [--label NAME]\n",
                     argv[0]);
        return 2;
    }
    std::string model_path = argv[1], ref_path = argv[2], json_path, label;
    bool rt = false;
    size_t steps = 100000, block = 1000;
    double paced_s = 0.0;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::stoul(argv[++i]);
        else if (!std::strcmp(argv[i], "--block") && i + 1 < argc) block = std::stoul(argv[++i]);
        else if (!std::strcmp(argv[i], "--paced-seconds") && i + 1 < argc) paced_s = std::stod(argv[++i]);
        else if (!std::strcmp(argv[i], "--rt")) rt = true;
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!std::strcmp(argv[i], "--label") && i + 1 < argc) label = argv[++i];
    }
#ifdef __APPLE__
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    auto ns_to_abs = [&](double ns) { return uint64_t(ns * tb.denom / tb.numer); };
    if (rt) {
        thread_time_constraint_policy_data_t pol;
        pol.period = uint32_t(ns_to_abs(20e6));
        pol.computation = uint32_t(ns_to_abs(1e6));
        pol.constraint = uint32_t(ns_to_abs(2e6));
        pol.preemptible = 1;
        if (thread_policy_set(mach_thread_self(), THREAD_TIME_CONSTRAINT_POLICY, (thread_policy_t)&pol,
                              THREAD_TIME_CONSTRAINT_POLICY_COUNT) != KERN_SUCCESS) {
            std::fprintf(stderr, "could not set real-time thread policy\n");
            return 1;
        }
    }
#else
    if (rt) {
        std::fprintf(stderr, "--rt is macOS-only (use SCHED_FIFO + clock_nanosleep on Linux)\n");
        return 1;
    }
#endif
    auto dec = cbps::load(model_path);
    Ref ref = load_ref(ref_path);
    if (ref.U != dec->n_units) {
        std::fprintf(stderr, "unit mismatch: model %u ref %u\n", dec->n_units, ref.U);
        return 1;
    }
    if (label.empty()) label = dec->name();

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

    // clock resolution: smallest non-zero difference between consecutive reads
    double res_us = 1e9;
    for (int i = 0; i < 100000; ++i) {
        auto a = clk::now(), b = clk::now();
        while (b == a) b = clk::now();
        res_us = std::min(res_us, us(b - a));
    }

    // 2a. batched compute latency
    volatile float sink = 0.f;
    dec->reset();
    for (size_t i = 0; i < 2000; ++i) dec->step(&ref.X[size_t(i % ref.T) * ref.U], out);  // warm-up
    std::vector<double> per_step_block;
    for (size_t s = 0; s + block <= steps; s += block) {
        auto a = clk::now();
        for (size_t i = s; i < s + block; ++i) dec->step(&ref.X[size_t(i % ref.T) * ref.U], out);
        auto b = clk::now();
        sink = sink + out[0];
        per_step_block.push_back(us(b - a) / double(block));
    }
    double batch_mean = 0;
    for (double v : per_step_block) batch_mean += v;
    batch_mean /= per_step_block.size();

    // 2b. per-call timing for the tail
    std::vector<double> call(steps);
    for (size_t i = 0; i < steps; ++i) {
        const float* x = &ref.X[size_t(i % ref.T) * ref.U];
        auto a = clk::now();
        dec->step(x, out);
        auto b = clk::now();
        sink = sink + out[0];
        call[i] = us(b - a);
    }

    // 3. paced 20 ms loop
    auto paced = [&](bool spin) {
        Paced p;
        size_t ticks = size_t(paced_s / 0.02);
        auto next = clk::now() + std::chrono::milliseconds(20);
        for (size_t i = 0; i < ticks; ++i) {
            if (spin) {
                while (clk::now() < next) {
                }
            } else {
#ifdef __APPLE__
                if (rt) {
                    auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(next - clk::now()).count();
                    if (wait > 0) mach_wait_until(mach_absolute_time() + ns_to_abs(double(wait)));
                } else {
                    std::this_thread::sleep_until(next);
                }
#else
                std::this_thread::sleep_until(next);
#endif
            }
            auto woke = clk::now();
            dec->step(&ref.X[size_t(i % ref.T) * ref.U], out);
            auto done = clk::now();
            p.late.push_back(us(woke - next));
            p.total.push_back(us(done - next));
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

    std::printf("%-12s params=%-8zu check=%s (max rel err %.2e)  batched %.3f us/step (p99 of %zu-step blocks "
                "%.3f)  per-call p99.9 %.2f max %.1f us  [clock res %.3f us]\n",
                label.c_str(), dec->n_params(), ok ? "PASS" : "FAIL", rel, batch_mean, block,
                pct(per_step_block, 0.99), pct(call, 0.999), *std::max_element(call.begin(), call.end()), res_us);
    for (auto* p : {&ps, &pp})
        if (!p->late.empty())
            std::printf("             paced 20 ms x %zu (%s%s): lateness p50 %.0f p99 %.0f us, "
                        "wake+compute p99 %.0f max %.0f us, deadline misses %zu\n",
                        p->late.size(), p == &ps ? "sleep" : "spin", rt ? ", rt thread" : "", pct(p->late, 0.5),
                        pct(p->late, 0.99), pct(p->total, 0.99), *std::max_element(p->total.begin(), p->total.end()),
                        p->misses);
    if (!json_path.empty()) {
        FILE* f = std::fopen(json_path.c_str(), "w");
        std::fprintf(f,
                     "{\"name\": \"%s\", \"n_params\": %zu, \"check_pass\": %s, \"max_rel_err\": %.3e, "
                     "\"steps\": %zu, \"block\": %zu, \"clock_resolution_us\": %.4f, "
                     "\"batched_mean_us\": %.4f, \"batched_block_p50_us\": %.4f, \"batched_block_p99_us\": %.4f, "
                     "\"call_p50_us\": %.4f, \"call_p99_us\": %.4f, \"call_p999_us\": %.4f, \"call_max_us\": %.3f, "
                     "\"rt_thread\": %s",
                     label.c_str(), dec->n_params(), ok ? "true" : "false", rel, steps, block, res_us, batch_mean,
                     pct(per_step_block, 0.5), pct(per_step_block, 0.99), pct(call, 0.5), pct(call, 0.99),
                     pct(call, 0.999), *std::max_element(call.begin(), call.end()), rt ? "true" : "false");
        for (auto* p : {&ps, &pp})
            if (!p->late.empty())
                std::fprintf(f,
                             ", \"paced_%s\": {\"ticks\": %zu, \"lateness_p50_us\": %.1f, \"lateness_p99_us\": %.1f, "
                             "\"total_p99_us\": %.1f, \"total_max_us\": %.1f, \"deadline_misses\": %zu}",
                             p == &ps ? "sleep" : "spin", p->late.size(), pct(p->late, 0.5), pct(p->late, 0.99),
                             pct(p->total, 0.99), *std::max_element(p->total.begin(), p->total.end()), p->misses);
        std::fprintf(f, "}\n");
        std::fclose(f);
    }
    return ok ? 0 : 1;
}
