#include "config.hpp"
#include "engine.hpp"
#include "weights.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace squid;

static std::vector<int32_t> parse_csv_tokens(const std::string& s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

static QuantMode parse_mode(const std::string& s) {
    if (s == "exact") return QuantMode::Exact;
    if (s == "fp8")   return QuantMode::FP8;
    if (s == "nvfp4") return QuantMode::NVFP4;
    fprintf(stderr, "Unknown mode '%s' (expected exact|fp8|nvfp4)\n", s.c_str());
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage:\n"
            "  squid5090 generate <model_dir> --tokens 1,2,3 [--mode exact|fp8|nvfp4] [--mtp 0..7] [--max-tokens 128] [--ctx 8192] [--no-graph]\n"
            "  squid5090 bench    <model_dir> --tokens 1,2,3 [--mode exact|fp8|nvfp4] [--mtp 0..7] [--max-tokens 128] [--runs 6] [--ctx 8192] [--no-graph]\n"
            "  squid5090 verify   <model_dir> --tokens 1,2,3 [--mode exact|fp8|nvfp4]\n");
        return 1;
    }

    std::string cmd = argv[1];
    std::string model_dir = argv[2];
    std::string tokens_str = "1,2,3";
    std::string tokens_file;
    std::string mode_str = "exact";
    std::string kv_str = "auto";
    int mtp = 0;
    int max_tokens = 128;
    int runs = 6;
    int ctx = 8192;
    int sim_pos = 0;
    bool use_graphs = true;
    bool embed_on_host = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) tokens_str = argv[++i];
        else if (a == "--tokens-file" && i + 1 < argc) tokens_file = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode_str = argv[++i];
        else if (a == "--kv" && i + 1 < argc) kv_str = argv[++i];
        else if (a == "--mtp" && i + 1 < argc) mtp = std::atoi(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--sim-pos" && i + 1 < argc) sim_pos = std::atoi(argv[++i]);
        else if (a == "--embed-host") embed_on_host = true;
        else if (a == "--no-graph") use_graphs = false;
    }

    QuantMode qmode = parse_mode(mode_str);
    KVFormat kv_fmt = KVFormat::BF16;
    if (kv_str == "bf16") kv_fmt = KVFormat::BF16;
    else if (kv_str == "fp8") kv_fmt = KVFormat::FP8;
    else if (kv_str == "q4") kv_fmt = KVFormat::Q4;
    else { // auto
        if (qmode == QuantMode::Exact) {
            kv_fmt = KVFormat::BF16;
        } else {
            kv_fmt = (ctx > 458752) ? KVFormat::Q4 : KVFormat::FP8;
        }
    }
    if (ctx > 524288 && qmode != QuantMode::Exact) {
        embed_on_host = true;
    }

    if (!tokens_file.empty()) {
        std::ifstream ifs(tokens_file);
        if (!ifs) {
            fprintf(stderr, "Failed to open --tokens-file '%s'\n", tokens_file.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << ifs.rdbuf();
        tokens_str = ss.str();
    }
    auto prompt = parse_csv_tokens(tokens_str);

    ModelWeights weights;
    if (!weights.load(model_dir, qmode,
                      /*load_mtp=*/mtp > 0 || cmd == "bench" || cmd == "longctx" || cmd == "serve-stdio",
                      embed_on_host)) {
        return 1;
    }

    Engine engine;
    if (!engine.init(&weights, ctx, use_graphs, kv_fmt)) return 1;

    if (cmd == "serve-stdio") {
        // Warm up CUDA graphs once before signaling READY
        std::vector<int32_t> warm = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        (void)engine.generate(warm, 8, mtp);
        printf("READY max_ctx=%d mode=%s kv=%s mtp=%d vram_gib=%.2f\n",
               ctx, mode_str.c_str(),
               kv_fmt == KVFormat::FP8 ? "fp8" : (kv_fmt == KVFormat::Q4 ? "q4" : "bf16"),
               mtp,
               (weights.total_vram_bytes + engine.runtime_vram_bytes()) / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line == "QUIT") break;
            if (line.rfind("GEN ", 0) == 0) {
                std::istringstream iss(line.substr(4));
                int req_max = 256;
                int req_mtp = mtp;
                std::string tok_csv;
                iss >> req_max >> req_mtp >> tok_csv;
                auto req_prompt = parse_csv_tokens(tok_csv);
                if (req_prompt.empty()) {
                    printf("ERR empty_prompt\n");
                    fflush(stdout);
                    continue;
                }
                if ((int)req_prompt.size() >= ctx) {
                    printf("ERR prompt_exceeds_ctx prompt=%zu max_ctx=%d\n", req_prompt.size(), ctx);
                    fflush(stdout);
                    continue;
                }
                if (req_mtp < 0) req_mtp = 0;
                if (req_mtp > MTP_MAX_DRAFTS) req_mtp = MTP_MAX_DRAFTS;
                if (req_max <= 0) req_max = 1;
                if ((int)req_prompt.size() + req_max > ctx) {
                    req_max = ctx - (int)req_prompt.size();
                }

                auto st = engine.generate(req_prompt, req_max, req_mtp, [&](int32_t tok) -> bool {
                    printf("T %d\n", tok);
                    fflush(stdout);
                    return true;
                });
                double dec_toks = std::max(0, st.gen_tokens - 1);
                double acc = (st.total_drafts > 0) ? (100.0 * st.total_accepted / st.total_drafts) : 0.0;
                double dl = (st.spec_steps > 0) ? (dec_toks / st.spec_steps) : 1.0;
                bool hit_eos = (!st.tokens.empty() && st.tokens.back() == EOS_TOKEN);
                printf("DONE %d %d %.2f %.2f %.1f %.2f %s\n",
                       st.prompt_tokens, st.gen_tokens, st.prefill_ms, st.decode_ms, acc, dl,
                       hit_eos ? "stop" : "length");
                fflush(stdout);
            } else {
                printf("ERR unknown_command\n");
                fflush(stdout);
            }
        }
        return 0;
    }

    if (cmd == "profile") {
        int B = (mtp > 0) ? (mtp + 1) : 1;
        engine.profile_step(prompt, B, mtp, sim_pos);
        return 0;
    }

    if (cmd == "longctx") {
        const int test_pos[] = {1024, 16384, 32768, 65536, 131072, 262144, 393216, 524288, 786432, 1048000};
        double total_vram = (weights.total_vram_bytes + engine.runtime_vram_bytes()) / (1024.0 * 1024.0 * 1024.0);
        printf("=== LONG-CONTEXT SCALING (mode=%s, kv=%s, max_ctx=%d, total_VRAM=%.2f GiB) ===\n",
               mode_str.c_str(), kv_fmt == KVFormat::FP8 ? "fp8" : (kv_fmt == KVFormat::Q4 ? "q4" : "bf16"),
               ctx, total_vram);
        printf("  %9s | %11s | %11s | %11s | %13s\n",
               "pos", "B=1 ms/tok", "B=1 tok/s", "B=4 ms/step", "B=4 @dl=2.5");
        for (int p : test_pos) {
            if (p + 256 >= ctx) continue;
            double ms1 = engine.bench_step_at_pos(p, 0, 24);
            double ms4 = engine.bench_step_at_pos(p, 3, 24);
            printf("  %9d | %8.2f ms | %7.1f t/s | %8.2f ms | %9.1f t/s\n",
                   p, ms1, 1000.0 / ms1, ms4, 2500.0 / ms4);
            fflush(stdout);
        }
        return 0;
    }

    if (cmd == "verify") {
        auto logits = engine.forward_logits(prompt);
        std::vector<int> idx(logits.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        printf("TOP10_LOGITS:");
        for (int i = 0; i < 10; ++i) printf(" %d:%.4f", idx[i], logits[idx[i]]);
        printf("\n");
        return 0;
    }

    if (cmd == "generate") {
        auto st = engine.generate(prompt, max_tokens, mtp);
        double dec_toks = std::max(0, st.gen_tokens - 1);
        double tps = (st.decode_ms > 0.0) ? (dec_toks * 1000.0 / st.decode_ms) : 0.0;
        double acc_pct = (st.total_drafts > 0) ? (100.0 * st.total_accepted / st.total_drafts) : 0.0;
        double dl = (st.spec_steps > 0) ? ((double)(st.gen_tokens - 1) / st.spec_steps) : 1.0;
        printf("[gen] prompt=%d gen=%d prefill=%.2fms decode=%.2fms speed=%.2f tok/s | mtp=%d acc=%.1f%% dl=%.2f\n",
               st.prompt_tokens, st.gen_tokens, st.prefill_ms, st.decode_ms, tps, mtp, acc_pct, dl);
        printf("TOKENS:");
        for (size_t i = 0; i < st.tokens.size(); ++i) {
            printf("%s%d", i ? "," : " ", st.tokens[i]);
        }
        printf("\n");
        return 0;
    }

    if (cmd == "bench") {
        // 1 warmup run (also captures CUDA Graph)
        (void)engine.generate(prompt, std::min(32, max_tokens), mtp);

        std::vector<double> speeds;
        double last_acc = 0.0, last_dl = 1.0, last_prefill = 0.0;
        std::vector<int32_t> sample_toks;
        for (int r = 0; r < runs; ++r) {
            auto st = engine.generate(prompt, max_tokens, mtp);
            double dec_toks = std::max(0, st.gen_tokens - 1);
            double tps = (st.decode_ms > 0.0) ? (dec_toks * 1000.0 / st.decode_ms) : 0.0;
            double acc = (st.total_drafts > 0) ? (100.0 * st.total_accepted / st.total_drafts) : 0.0;
            double dl = (st.spec_steps > 0) ? (dec_toks / st.spec_steps) : 1.0;
            speeds.push_back(tps);
            last_acc = acc;
            last_dl = dl;
            last_prefill = st.prefill_ms;
            if (r == 0) sample_toks = st.tokens;
            printf("  run %d/%d: gen=%d decode=%.2fms speed=%.2f tok/s (acc=%.1f%% dl=%.2f)\n",
                   r + 1, runs, st.gen_tokens, st.decode_ms, tps, acc, dl);
        }
        std::sort(speeds.begin(), speeds.end());
        double med = (speeds.size() % 2 == 1)
            ? speeds[speeds.size() / 2]
            : 0.5 * (speeds[speeds.size() / 2 - 1] + speeds[speeds.size() / 2]);
        printf("MEDIAN: %.2f tok/s | min=%.2f max=%.2f | prefill=%.2fms | acc=%.1f%% dl=%.2f | mode=%s mtp=%d graph=%d\n",
               med, speeds.front(), speeds.back(), last_prefill, last_acc, last_dl,
               mode_str.c_str(), mtp, use_graphs ? 1 : 0);
        printf("TOKENS:");
        for (size_t i = 0; i < sample_toks.size(); ++i) printf("%s%d", i ? "," : " ", sample_toks[i]);
        printf("\n");
        return 0;
    }

    fprintf(stderr, "Unknown command '%s'\n", cmd.c_str());
    return 1;
}
