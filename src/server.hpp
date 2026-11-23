#pragma once

#include "engine.hpp"
#include "tokenizer.hpp"

#include <string>

namespace squid {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8000;
    std::string model_name = "inkling-2-mini";
    std::string api_key;
    std::string default_effort = "none";
    int default_mtp = 3;
    int max_ctx = 65536;
    bool raw_thinking = false;
};

// Runs the native C++20 POSIX HTTP/1.1 + SSE OpenAI-compatible API server until SIGINT/SIGTERM.
int run_openai_server(Engine& engine, const Tokenizer& tokenizer, const ServerConfig& cfg);

} // namespace squid
