#include "config.hpp"
#include "engine.hpp"
#include "json.hpp"
#include "server.hpp"
#include "tokenizer.hpp"
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
            "squid5090 — Native C++/CUDA Blackwell Inference Engine & OpenAI Server for Inkling-2-Mini\n\n"
            "Usage:\n"
            "  squid5090 serve    <model_dir> [--host 0.0.0.0] [--port 8000] [--mode nvfp4|fp8|exact] [--kv auto|bf16|fp8|q4] [--ctx 65536] [--mtp 3] [--model-name inkling-2-mini] [--effort none] [--api-key KEY]\n"
            "  squid5090 generate <model_dir> (--prompt TEXT | --tokens CSV | --tokens-file PATH) [--mode nvfp4] [--kv auto] [--mtp 3] [--max-tokens 512] [--ctx 8192]\n"
            "  squid5090 bench    <model_dir> (--prompt TEXT | --tokens CSV) [--mode nvfp4] [--kv auto] [--mtp 3] [--max-tokens 512] [--runs 6] [--ctx 8192]\n"
            "  squid5090 longctx  <model_dir> [--mode nvfp4] [--kv fp8|q4|bf16] [--ctx 1048576]\n"
            "  squid5090 profile  <model_dir> [--mode nvfp4] [--kv auto] [--mtp 3] [--ctx 8192] [--sim-pos 0]\n"
            "  squid5090 verify   <model_dir> --tokens 1,2,3 [--mode exact|fp8|nvfp4]\n"
            "  squid5090 tokenize <model_dir> --prompt TEXT [--raw]\n");
        return 1;
    }

    std::string cmd = argv[1];
    std::string model_dir = argv[2];
    std::string tokens_str = "1,2,3";
    std::string tokens_file;
    std::string prompt_text;
    std::string mode_str = (cmd == "serve") ? "nvfp4" : "exact";
    std::string kv_str = "auto";
    std::string host = "0.0.0.0";
    std::string model_name = "inkling-2-mini";
    std::string effort = "none";
    std::string api_key = std::getenv("SQUID_API_KEY") ? std::getenv("SQUID_API_KEY") : "";
    int port = 8000;
    int mtp = (cmd == "serve") ? 3 : 0;
    int max_tokens = 128;
    int runs = 6;
    int ctx = (cmd == "serve") ? 65536 : 8192;
    int sim_pos = 0;
    bool use_graphs = true;
    bool embed_on_host = false;
    bool raw_prompt = false;
    bool raw_thinking = false;
    bool has_tokens_arg = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) { tokens_str = argv[++i]; has_tokens_arg = true; }
        else if (a == "--tokens-file" && i + 1 < argc) { tokens_file = argv[++i]; has_tokens_arg = true; }
        else if (a == "--prompt" && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode_str = argv[++i];
        else if (a == "--kv" && i + 1 < argc) kv_str = argv[++i];
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--model-name" && i + 1 < argc) model_name = argv[++i];
        else if (a == "--effort" && i + 1 < argc) effort = argv[++i];
        else if (a == "--api-key" && i + 1 < argc) api_key = argv[++i];
        else if (a == "--mtp" && i + 1 < argc) mtp = std::atoi(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--sim-pos" && i + 1 < argc) sim_pos = std::atoi(argv[++i]);
        else if (a == "--embed-host") embed_on_host = true;
        else if (a == "--no-graph") use_graphs = false;
        else if (a == "--raw") raw_prompt = true;
        else if (a == "--raw-thinking") raw_thinking = true;
    }

    // Fast tokenizer-only diagnostic command (no GPU allocation needed)
    if (cmd == "tokenize") {
        Tokenizer tok;
        if (!tok.load(model_dir)) return 1;
        std::string input = prompt_text;
        if (!raw_prompt) {
            Json msgs = Json::array();
            Json m = Json::object();
            m["role"] = "user";
            m["content"] = prompt_text;
            msgs.push_back(std::move(m));
            Json eff(effort);
            input = tok.render_chat_template(msgs, nullptr, &eff);
        }
        auto ids = tok.encode(input, true);
        printf("IDS:");
        for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? "," : " ", ids[i]);
        printf("\n");
        return 0;
    }

    bool need_tokenizer = (cmd == "serve" || !prompt_text.empty());
    Tokenizer tokenizer;
    if (need_tokenizer && !tokenizer.load(model_dir)) {
        return 1;
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

    std::vector<int32_t> prompt;
    if (!prompt_text.empty() && !has_tokens_arg) {
        std::string rendered = prompt_text;
        if (!raw_prompt) {
            Json msgs = Json::array();
            Json m = Json::object();
            m["role"] = "user";
            m["content"] = prompt_text;
            msgs.push_back(std::move(m));
            Json eff(effort);
            rendered = tokenizer.render_chat_template(msgs, nullptr, &eff);
        }
        prompt = tokenizer.encode(rendered, true);
    } else {
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
        prompt = parse_csv_tokens(tokens_str);
    }

    ModelWeights weights;
    bool load_mtp = (mtp > 0 || cmd == "bench" || cmd == "longctx" || cmd == "serve");
    if (!weights.load(model_dir, qmode, load_mtp, embed_on_host)) {
        return 1;
    }

    Engine engine;
    if (!engine.init(&weights, ctx, use_graphs, kv_fmt)) return 1;

    if (cmd == "serve") {
        ServerConfig scfg;
        scfg.host = host;
        scfg.port = port;
        scfg.model_name = model_name;
        scfg.api_key = api_key;
        scfg.default_effort = effort;
        scfg.default_mtp = mtp;
        scfg.max_ctx = ctx;
        scfg.raw_thinking = raw_thinking;
        return run_openai_server(engine, tokenizer, scfg);
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
        if (need_tokenizer) {
            std::string decoded = tokenizer.decode(st.tokens, false);
            printf("\n--- DECODED OUTPUT ---\n%s\n----------------------\n", decoded.c_str());
        }
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
        if (need_tokenizer) {
            std::string decoded = tokenizer.decode(sample_toks, false);
            printf("\n--- DECODED OUTPUT ---\n%s\n----------------------\n", decoded.c_str());
        }
        return 0;
    }

    fprintf(stderr, "Unknown command '%s'\n", cmd.c_str());
    return 1;
}
