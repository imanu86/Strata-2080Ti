// Exercise #500 through the actual pool, including serial/parallel transitions and >96 jobs.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;
struct Act { alignas(64) uint8_t data[c::kNativeActBytes]; };
int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) { std::fprintf(stderr, "usage: pool_quant_parity shard1.gguf [ff=640]\n"); return 2; }
    const int ff = argc == 3 ? std::atoi(argv[2]) : c::FF;
    if (ff <= 0 || ff > c::FF || ff % 64) return 2;
    try {
        strata::GgufFile gguf(argv[1]);
        int cases = 0;
        for (int layer : {0, 20}) {
            const strata::TensorInfo* tensor[3] = {};
            const char* role[] = {"gate", "up", "down"};
            for (const auto& t : gguf.tensors())
                for (int r = 0; r < 3; ++r)
                    if (t.name == "blk." + std::to_string(layer) + ".ffn_" + role[r] + "_exps.weight") tensor[r] = &t;
            if (!tensor[0] || !tensor[1] || !tensor[2]) return 2;
            c::NativeFmt f; std::string err;
            if (!c::native_fmt(tensor[0]->type, tensor[2]->type, c::H, ff, f, err)) {
                std::fprintf(stderr, "%s\n", err.c_str()); return 2;
            }
            c::NativeFmt original;
            if (!c::native_fmt(tensor[0]->type, tensor[2]->type, c::H, c::FF, original, err)) return 2;
            std::vector<uint8_t> blobs(8 * f.bytes);
            for (int e = 0; e < 8; ++e) {
                auto* p = blobs.data() + e * f.bytes;
                std::memcpy(p, gguf.tensor_data(*tensor[0]) + e*original.up_off, f.up_off);
                std::memcpy(p+f.up_off, gguf.tensor_data(*tensor[1]) + e*original.up_off, f.up_off);
                for (int row = 0; row < c::H; ++row)
                    std::memcpy(p+f.down_off+row*f.d_row,
                                gguf.tensor_data(*tensor[2]) + e*(original.bytes-original.down_off)+row*original.d_row, f.d_row);
            }
            std::printf("layer %d native gu=%d down=%d, %zu bytes/expert\n", layer, f.gu_type, f.d_type, f.bytes);
            for (bool host : {false, true}) {
                // Two workers suffice to exercise publication and keep this a small correctness test.
                c::ExpertPool pool(2, false, host);
                for (int n : {1, 3, 9, 97}) {
                    std::vector<c::ExpertJobMulti> jobs(n);
                    std::vector<Act> acts(n*c::MAXT);
                    std::vector<float> out((size_t)n*c::MAXT*c::H), ref(out.size());
                    std::vector<float> x(c::H);
                    std::mt19937 rng(6917+n);
                    std::normal_distribution<float> rnd(0.f, .2f);
                    for (int e = 0; e < n; ++e) {
                        jobs[e].blob = blobs.data() + (e % 8)*f.bytes;
                        jobs[e].nt = n == 1 ? c::MAXT : 1 + e % c::MAXT;
                        for (int t = 0; t < jobs[e].nt; ++t) {
                            for (auto& v : x) v = rnd(rng);
                            auto& a = acts[e*c::MAXT+t];
                            c::native_quant_act(f, x.data(), a.data);
                            jobs[e].nact[t] = a.data;
                            jobs[e].out[t] = out.data()+((size_t)e*c::MAXT+t)*c::H;
                        }
                    }
                    _putenv_s("STRATA_PARALLEL_INTERMEDIATE_QUANT", "0");
                    pool.run_split_multi_native(f, jobs.data(), n);
                    ref = out;
                    for (int repeat = 0; repeat < 4; ++repeat) {
                        _putenv_s("STRATA_PARALLEL_INTERMEDIATE_QUANT", repeat % 2 ? "0" : "1");
                        std::fill(out.begin(), out.end(), NAN);
                        pool.run_split_multi_native(f, jobs.data(), n);
                        for (int e = 0; e < n; ++e)
                            for (int t = 0; t < jobs[e].nt; ++t) {
                                const size_t start = ((size_t)e*c::MAXT+t)*c::H;
                                for (int d = 0; d < c::H; ++d) {
                                    if (!std::isfinite(out[start+d]) ||
                                        std::memcmp(&out[start+d], &ref[start+d], sizeof(float))) {
                                        std::fprintf(stderr, "FAIL l=%d host=%d n=%d repeat=%d e=%d t=%d d=%d\n",
                                                     layer, host, n, repeat, e, t, d); return 1;
                                    }
                                }
                            }
                        ++cases;
                    }
                }
            }
        }
        std::printf("PASS: %d native pool cases, outputs finite and bitwise serial; tokens 1..8, batch boundary 96/97\n", cases);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    _putenv_s("STRATA_PARALLEL_INTERMEDIATE_QUANT", "");
    return 0;
}
