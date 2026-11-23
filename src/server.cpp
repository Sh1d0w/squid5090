#include "server.hpp"
#include "json.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace squid {

namespace {

std::atomic<bool> g_stop_requested{false};
std::atomic<int>  g_listen_fd{-1};

void handle_signal(int) {
    g_stop_requested.store(true);
    int fd = g_listen_fd.exchange(-1);
    if (fd >= 0) ::close(fd);
}

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += (size_t)n;
    }
    return true;
}

bool send_str(int fd, const std::string& s) {
    return send_all(fd, s.data(), s.size());
}

std::string random_hex(size_t nchars) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static constexpr char hex[] = "0123456789abcdef";
    std::string out(nchars, '0');
    for (size_t i = 0; i < nchars; ++i) out[i] = hex[rng() & 15];
    return out;
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string trim_ws(std::string_view sv) {
    size_t a = 0, b = sv.size();
    while (a < b && (sv[a] == ' ' || sv[a] == '\t' || sv[a] == '\r' || sv[a] == '\n')) ++a;
    while (b > a && (sv[b - 1] == ' ' || sv[b - 1] == '\t' || sv[b - 1] == '\r' || sv[b - 1] == '\n')) --b;
    return std::string(sv.substr(a, b - a));
}

// Compute length of complete valid UTF-8 prefix (holding back any incomplete trailing multi-byte char)
size_t valid_utf8_prefix_len(std::string_view s) {
    size_t n = s.size();
    if (n == 0) return 0;
    // Check last up to 3 bytes for an incomplete multi-byte leader
    size_t check = std::min<size_t>(n, 3);
    for (size_t back = 1; back <= check; ++back) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0x80) == 0) return n; // ASCII byte ends cleanly
        if ((c & 0xC0) == 0xC0) {      // UTF-8 leading byte
            size_t expected = ((c & 0xE0) == 0xC0) ? 2
                            : ((c & 0xF0) == 0xE0) ? 3
                            : ((c & 0xF8) == 0xF0) ? 4 : 1;
            return (back < expected) ? (n - back) : n;
        }
    }
    return n;
}

std::vector<std::string> parse_stops(const Json* v) {
    std::vector<std::string> out;
    if (!v || v->is_null()) return out;
    if (v->is_string() && !v->as_string().empty()) {
        out.push_back(v->as_string());
    } else if (v->is_array()) {
        for (const auto& e : v->arr) {
            if (e.is_string() && !e.as_string().empty()) out.push_back(e.as_string());
        }
    }
    return out;
}

size_t find_first_stop(const std::string& text, const std::vector<std::string>& stops) {
    size_t best = std::string::npos;
    for (const auto& s : stops) {
        size_t pos = text.find(s);
        if (pos != std::string::npos && pos < best) best = pos;
    }
    return best;
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
};

bool read_http_request(int fd, HttpRequest& req) {
    std::string buf;
    buf.reserve(4096);
    char tmp[8192];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > 64 * 1024 && header_end == std::string::npos) return false;
    }

    std::string_view hdr_block(buf.data(), header_end);
    size_t line_end = hdr_block.find("\r\n");
    std::string_view req_line = (line_end == std::string_view::npos) ? hdr_block : hdr_block.substr(0, line_end);

    // Parse "METHOD /path HTTP/1.1"
    size_t sp1 = req_line.find(' ');
    if (sp1 == std::string_view::npos) return false;
    size_t sp2 = req_line.find(' ', sp1 + 1);
    if (sp2 == std::string_view::npos) return false;
    req.method = std::string(req_line.substr(0, sp1));
    std::string full_path(req_line.substr(sp1 + 1, sp2 - sp1 - 1));
    size_t qpos = full_path.find('?');
    req.path = (qpos == std::string::npos) ? full_path : full_path.substr(0, qpos);

    size_t pos = (line_end == std::string_view::npos) ? hdr_block.size() : (line_end + 2);
    while (pos < hdr_block.size()) {
        size_t next = hdr_block.find("\r\n", pos);
        std::string_view line = (next == std::string_view::npos) ? hdr_block.substr(pos) : hdr_block.substr(pos, next - pos);
        size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            req.headers[to_lower_ascii(trim_ws(line.substr(0, colon)))] = trim_ws(line.substr(colon + 1));
        }
        if (next == std::string_view::npos) break;
        pos = next + 2;
    }

    // Honor Expect: 100-continue (curl uses this for large prompts!)
    auto exp_it = req.headers.find("expect");
    if (exp_it != req.headers.end() && to_lower_ascii(exp_it->second) == "100-continue") {
        send_str(fd, "HTTP/1.1 100 Continue\r\n\r\n");
    }

    size_t content_len = 0;
    auto cl_it = req.headers.find("content-length");
    if (cl_it != req.headers.end()) {
        content_len = (size_t)std::strtoull(cl_it->second.c_str(), nullptr, 10);
    }
    if (content_len > 64 * 1024 * 1024) return false; // 64 MB safety cap

    size_t body_start = header_end + 4;
    req.body.assign(buf.data() + body_start, buf.size() - body_start);
    while (req.body.size() < content_len) {
        size_t need = std::min<size_t>(sizeof(tmp), content_len - req.body.size());
        ssize_t n = ::recv(fd, tmp, need, 0);
        if (n <= 0) return false;
        req.body.append(tmp, (size_t)n);
    }
    if (req.body.size() > content_len) req.body.resize(content_len);
    return true;
}

const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default:  return "OK";
    }
}

void send_json(int fd, int code, const Json& payload) {
    std::string body = payload.dump();
    std::string hdr =
        "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    send_str(fd, hdr);
    send_str(fd, body);
}

void send_error(int fd, int code, const std::string& msg, const char* type = "invalid_request_error") {
    Json err = Json::object();
    err["error"] = Json::object();
    err["error"]["message"] = msg;
    err["error"]["type"] = type;
    send_json(fd, code, err);
}

bool send_sse_headers(int fd) {
    return send_str(fd,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n");
}

bool send_sse_json(int fd, const Json& j) {
    std::string line = "data: " + j.dump() + "\n\n";
    return send_str(fd, line);
}

bool send_sse_done(int fd) {
    return send_str(fd, "data: [DONE]\n\n");
}

// Incremental Inkling-2-Mini chat section parser (thinking / text / tool_calls)
class InklingStreamParser {
public:
    InklingStreamParser(const Tokenizer& tok, bool raw_thinking)
        : tok_(tok), raw_thinking_(raw_thinking) {}

    void feed(int32_t id, std::string& delta_reasoning, std::string& delta_content) {
        delta_reasoning.clear();
        delta_content.clear();
        if (id == tok_.id_eos) return;

        if (raw_thinking_) {
            pending_bytes_ += tok_.decode_token(id, false);
            size_t ok = valid_utf8_prefix_len(pending_bytes_);
            if (ok > 0) {
                delta_content.assign(pending_bytes_.data(), ok);
                pending_bytes_.erase(0, ok);
                content_full += delta_content;
            }
            return;
        }

        if (id == tok_.id_msg_model) {
            section_ = Section::Header;
            pending_bytes_.clear();
            tool_buf_.clear();
            return;
        }
        if (id == tok_.id_thinking) {
            section_ = Section::Thinking;
            pending_bytes_.clear();
            return;
        }
        if (id == tok_.id_text) {
            section_ = Section::Text;
            pending_bytes_.clear();
            return;
        }
        if (tok_.id_tool_json >= 0 && id == tok_.id_tool_json) {
            section_ = Section::Tool;
            tool_buf_.clear();
            return;
        }
        if (id == tok_.id_end_msg) {
            if (section_ == Section::Tool && !tool_buf_.empty()) {
                try {
                    Json obj = json_parse(tool_buf_);
                    std::string fname = (obj.get("name") && obj.get("name")->is_string())
                                      ? obj.get("name")->as_string() : "";
                    const Json* args = obj.get("args") ? obj.get("args") : obj.get("arguments");
                    std::string args_str = args ? args->dump() : "{}";
                    Json tc = Json::object();
                    tc["id"] = "call_" + random_hex(12);
                    tc["type"] = "function";
                    tc["function"] = Json::object();
                    tc["function"]["name"] = fname;
                    tc["function"]["arguments"] = args_str;
                    tool_calls.push_back(std::move(tc));
                } catch (...) {}
                tool_buf_.clear();
            }
            section_ = Section::None;
            pending_bytes_.clear();
            return;
        }

        if (section_ == Section::Header) return;
        if (section_ == Section::Tool) {
            tool_buf_ += tok_.decode_token(id, true);
            return;
        }

        // Thinking, Text, or fallback None -> Text
        pending_bytes_ += tok_.decode_token(id, false);
        size_t ok = valid_utf8_prefix_len(pending_bytes_);
        if (ok == 0) return;
        std::string piece(pending_bytes_.data(), ok);
        pending_bytes_.erase(0, ok);
        if (section_ == Section::Thinking) {
            reasoning_full += piece;
            delta_reasoning = std::move(piece);
        } else {
            content_full += piece;
            delta_content = std::move(piece);
        }
    }

    std::string reasoning_full;
    std::string content_full;
    std::vector<Json> tool_calls;

private:
    enum class Section { None, Header, Thinking, Text, Tool };
    const Tokenizer& tok_;
    bool raw_thinking_;
    Section section_ = Section::None;
    std::string pending_bytes_;
    std::string tool_buf_;
};

class ServerContext {
public:
    Engine& engine;
    const Tokenizer& tok;
    ServerConfig cfg;
    std::mutex gpu_mu;

    void handle_client(int fd) {
        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        HttpRequest req;
        if (!read_http_request(fd, req)) {
            ::close(fd);
            return;
        }

        if (req.method == "OPTIONS") {
            send_str(fd,
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n");
            ::close(fd);
            return;
        }

        if (req.method == "GET" && req.path == "/health") {
            Json j = Json::object();
            j["status"] = "ok";
            j["model"] = cfg.model_name;
            j["max_context"] = cfg.max_ctx;
            j["mtp"] = cfg.default_mtp;
            send_json(fd, 200, j);
            ::close(fd);
            return;
        }

        if (!cfg.api_key.empty()) {
            auto it = req.headers.find("authorization");
            if (it == req.headers.end() || it->second != ("Bearer " + cfg.api_key)) {
                send_error(fd, 401, "Invalid API key", "authentication_error");
                ::close(fd);
                return;
            }
        }

        if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
            Json m = Json::object();
            m["id"] = cfg.model_name;
            m["object"] = "model";
            m["created"] = (int64_t)std::time(nullptr);
            m["owned_by"] = "squid5090";
            m["max_model_len"] = cfg.max_ctx;
            Json j = Json::object();
            j["object"] = "list";
            j["data"] = Json::array();
            j["data"].push_back(std::move(m));
            send_json(fd, 200, j);
            ::close(fd);
            return;
        }

        if (req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions")) {
            handle_chat(fd, req.body);
            ::close(fd);
            return;
        }

        if (req.method == "POST" && (req.path == "/v1/completions" || req.path == "/completions")) {
            handle_completions(fd, req.body);
            ::close(fd);
            return;
        }

        send_error(fd, 404, "Unknown endpoint: " + req.method + " " + req.path);
        ::close(fd);
    }

    void handle_chat(int fd, const std::string& body) {
        Json req;
        try {
            req = json_parse(body.empty() ? "{}" : body);
        } catch (const std::exception& e) {
            send_error(fd, 400, std::string("Invalid JSON: ") + e.what());
            return;
        }

        const Json* messages = req.get("messages");
        if (!messages || !messages->is_array() || messages->arr.empty()) {
            send_error(fd, 400, "'messages' must be a non-empty array");
            return;
        }

        int max_tokens = 1024;
        if (const Json* mt = req.get("max_completion_tokens")) max_tokens = (int)mt->as_int(1024);
        else if (const Json* mt2 = req.get("max_tokens")) max_tokens = (int)mt2->as_int(1024);
        if (max_tokens < 1) max_tokens = 1;

        bool stream = req.get("stream") ? req.get("stream")->as_bool(false) : false;
        bool include_usage = false;
        if (const Json* so = req.get("stream_options")) {
            if (const Json* iu = so->get("include_usage")) include_usage = iu->as_bool(false);
        }
        auto stops = parse_stops(req.get("stop"));
        size_t max_stop_len = 0;
        for (const auto& s : stops) max_stop_len = std::max(max_stop_len, s.size());

        int req_mtp = req.get("mtp") ? (int)req.get("mtp")->as_int(cfg.default_mtp) : cfg.default_mtp;
        req_mtp = std::max(0, std::min(MTP_MAX_DRAFTS, req_mtp));

        std::string req_model = (req.get("model") && req.get("model")->is_string())
                              ? req.get("model")->as_string() : cfg.model_name;

        Json def_effort(cfg.default_effort);
        const Json* effort = req.get("reasoning_effort") ? req.get("reasoning_effort") : &def_effort;
        std::string rendered = tok.render_chat_template(*messages, req.get("tools"), effort);
        auto prompt_ids = tok.encode(rendered, true);

        if ((int)prompt_ids.size() >= cfg.max_ctx) {
            send_error(fd, 400, "Prompt tokens (" + std::to_string(prompt_ids.size()) +
                                ") exceed max_ctx (" + std::to_string(cfg.max_ctx) + ")");
            return;
        }
        if ((int)prompt_ids.size() + max_tokens > cfg.max_ctx) {
            max_tokens = cfg.max_ctx - (int)prompt_ids.size();
        }

        std::string req_id = "chatcmpl-" + random_hex(24);
        int64_t created = (int64_t)std::time(nullptr);
        InklingStreamParser parser(tok, cfg.raw_thinking);

        if (!stream) {
            bool hit_stop = false;
            GenStats stats;
            {
                std::lock_guard<std::mutex> lk(gpu_mu);
                stats = engine.generate(prompt_ids, max_tokens, req_mtp, [&](int32_t tid) -> bool {
                    std::string dr, dc;
                    parser.feed(tid, dr, dc);
                    if (!stops.empty()) {
                        size_t idx = find_first_stop(parser.content_full, stops);
                        if (idx != std::string::npos) {
                            parser.content_full.resize(idx);
                            hit_stop = true;
                            return false; // Stop GPU immediately!
                        }
                    }
                    return true;
                });
            }

            bool hit_eos = (!stats.tokens.empty() && stats.tokens.back() == EOS_TOKEN);
            std::string finish = !parser.tool_calls.empty() ? "tool_calls"
                               : (hit_stop || hit_eos) ? "stop" : "length";
            double dec_toks = std::max(0, stats.gen_tokens - 1);
            double tps = (stats.decode_ms > 0.0) ? (dec_toks * 1000.0 / stats.decode_ms) : 0.0;
            double acc = (stats.total_drafts > 0) ? (100.0 * stats.total_accepted / stats.total_drafts) : 0.0;
            double dl = (stats.spec_steps > 0) ? (dec_toks / stats.spec_steps) : 1.0;

            Json msg = Json::object();
            msg["role"] = "assistant";
            if (!parser.content_full.empty() || parser.tool_calls.empty()) {
                msg["content"] = parser.content_full;
            } else {
                msg["content"] = nullptr;
            }
            if (!parser.reasoning_full.empty()) msg["reasoning_content"] = parser.reasoning_full;
            if (!parser.tool_calls.empty()) {
                Json tcs = Json::array();
                for (auto& tc : parser.tool_calls) tcs.push_back(std::move(tc));
                msg["tool_calls"] = std::move(tcs);
            }

            Json choice = Json::object();
            choice["index"] = 0;
            choice["message"] = std::move(msg);
            choice["finish_reason"] = finish;

            Json resp = Json::object();
            resp["id"] = req_id;
            resp["object"] = "chat.completion";
            resp["created"] = created;
            resp["model"] = req_model;
            resp["choices"] = Json::array();
            resp["choices"].push_back(std::move(choice));
            resp["usage"] = Json::object();
            resp["usage"]["prompt_tokens"] = stats.prompt_tokens;
            resp["usage"]["completion_tokens"] = stats.gen_tokens;
            resp["usage"]["total_tokens"] = stats.prompt_tokens + stats.gen_tokens;
            resp["timings"] = Json::object();
            resp["timings"]["prefill_ms"] = stats.prefill_ms;
            resp["timings"]["decode_ms"] = stats.decode_ms;
            resp["timings"]["tokens_per_second"] = tps;
            resp["timings"]["spec_acc_pct"] = acc;
            resp["timings"]["spec_dl"] = dl;

            printf("[http] POST /v1/chat/completions prompt=%d gen=%d prefill=%.1fms decode=%.1fms (%.1f tok/s, dl=%.2f)\n",
                   stats.prompt_tokens, stats.gen_tokens, stats.prefill_ms, stats.decode_ms, tps, dl);
            fflush(stdout);
            send_json(fd, 200, resp);
            return;
        }

        // Streaming SSE response
        if (!send_sse_headers(fd)) return;

        auto make_chunk = [&](Json delta, const Json& finish_reason) -> Json {
            Json ch = Json::object();
            ch["index"] = 0;
            ch["delta"] = std::move(delta);
            ch["finish_reason"] = finish_reason;
            Json c = Json::object();
            c["id"] = req_id;
            c["object"] = "chat.completion.chunk";
            c["created"] = created;
            c["model"] = req_model;
            c["choices"] = Json::array();
            c["choices"].push_back(std::move(ch));
            return c;
        };

        {
            Json d0 = Json::object();
            d0["role"] = "assistant";
            if (!send_sse_json(fd, make_chunk(std::move(d0), Json(nullptr)))) return;
        }

        bool hit_stop = false;
        bool conn_ok = true;
        size_t emitted_len = 0;
        GenStats stats;
        {
            std::lock_guard<std::mutex> lk(gpu_mu);
            stats = engine.generate(prompt_ids, max_tokens, req_mtp, [&](int32_t tid) -> bool {
                std::string dr, dc;
                parser.feed(tid, dr, dc);
                if (!dr.empty()) {
                    Json d = Json::object();
                    d["reasoning_content"] = dr;
                    if (!send_sse_json(fd, make_chunk(std::move(d), Json(nullptr)))) {
                        conn_ok = false;
                        return false; // Client disconnected -> stop GPU immediately!
                    }
                }
                if (!dc.empty()) {
                    if (!stops.empty()) {
                        size_t idx = find_first_stop(parser.content_full, stops);
                        if (idx != std::string::npos) {
                            if (idx > emitted_len) {
                                Json d = Json::object();
                                d["content"] = parser.content_full.substr(emitted_len, idx - emitted_len);
                                send_sse_json(fd, make_chunk(std::move(d), Json(nullptr)));
                            }
                            emitted_len = idx;
                            hit_stop = true;
                            return false; // Stop GPU immediately!
                        }
                        size_t hold = (max_stop_len > 1) ? (max_stop_len - 1) : 0;
                        if (parser.content_full.size() > emitted_len + hold) {
                            size_t send_end = parser.content_full.size() - hold;
                            Json d = Json::object();
                            d["content"] = parser.content_full.substr(emitted_len, send_end - emitted_len);
                            if (!send_sse_json(fd, make_chunk(std::move(d), Json(nullptr)))) {
                                conn_ok = false;
                                return false;
                            }
                            emitted_len = send_end;
                        }
                    } else {
                        Json d = Json::object();
                        d["content"] = dc;
                        if (!send_sse_json(fd, make_chunk(std::move(d), Json(nullptr)))) {
                            conn_ok = false;
                            return false;
                        }
                        emitted_len = parser.content_full.size();
                    }
                }
                return true;
            });
        }

        if (!conn_ok) return;

        // Flush any held-back non-stop suffix
        if (!stops.empty() && !hit_stop && emitted_len < parser.content_full.size()) {
            Json d = Json::object();
            d["content"] = parser.content_full.substr(emitted_len);
            if (!send_sse_json(fd, make_chunk(std::move(d), Json(nullptr)))) return;
        }

        if (!parser.tool_calls.empty()) {
            Json d = Json::object();
            d["tool_calls"] = Json::array();
            for (size_t i = 0; i < parser.tool_calls.size(); ++i) {
                Json tc = parser.tool_calls[i];
                tc["index"] = (int)i;
                d["tool_calls"].push_back(std::move(tc));
            }
            if (!send_sse_json(fd, make_chunk(std::move(d), Json(nullptr)))) return;
        }

        bool hit_eos = (!stats.tokens.empty() && stats.tokens.back() == EOS_TOKEN);
        std::string finish = !parser.tool_calls.empty() ? "tool_calls"
                           : (hit_stop || hit_eos) ? "stop" : "length";
        Json fin = make_chunk(Json::object(), Json(finish));
        if (include_usage) {
            fin["usage"] = Json::object();
            fin["usage"]["prompt_tokens"] = stats.prompt_tokens;
            fin["usage"]["completion_tokens"] = stats.gen_tokens;
            fin["usage"]["total_tokens"] = stats.prompt_tokens + stats.gen_tokens;
        }
        send_sse_json(fd, fin);
        send_sse_done(fd);

        double dec_toks = std::max(0, stats.gen_tokens - 1);
        double tps = (stats.decode_ms > 0.0) ? (dec_toks * 1000.0 / stats.decode_ms) : 0.0;
        double dl = (stats.spec_steps > 0) ? (dec_toks / stats.spec_steps) : 1.0;
        printf("[http] POST /v1/chat/completions (stream) prompt=%d gen=%d prefill=%.1fms decode=%.1fms (%.1f tok/s, dl=%.2f)\n",
               stats.prompt_tokens, stats.gen_tokens, stats.prefill_ms, stats.decode_ms, tps, dl);
        fflush(stdout);
    }

    void handle_completions(int fd, const std::string& body) {
        Json req;
        try {
            req = json_parse(body.empty() ? "{}" : body);
        } catch (const std::exception& e) {
            send_error(fd, 400, std::string("Invalid JSON: ") + e.what());
            return;
        }

        std::string prompt;
        if (const Json* pr = req.get("prompt")) {
            if (pr->is_string()) prompt = pr->as_string();
            else if (pr->is_array() && !pr->arr.empty() && pr->arr[0].is_string()) prompt = pr->arr[0].as_string();
        }
        if (prompt.empty()) {
            send_error(fd, 400, "'prompt' must be a non-empty string");
            return;
        }

        int max_tokens = req.get("max_tokens") ? std::max(1, (int)req.get("max_tokens")->as_int(256)) : 256;
        bool stream = req.get("stream") ? req.get("stream")->as_bool(false) : false;
        bool echo = req.get("echo") ? req.get("echo")->as_bool(false) : false;
        auto stops = parse_stops(req.get("stop"));
        size_t max_stop_len = 0;
        for (const auto& s : stops) max_stop_len = std::max(max_stop_len, s.size());

        int req_mtp = req.get("mtp") ? (int)req.get("mtp")->as_int(cfg.default_mtp) : cfg.default_mtp;
        req_mtp = std::max(0, std::min(MTP_MAX_DRAFTS, req_mtp));

        std::string req_model = (req.get("model") && req.get("model")->is_string())
                              ? req.get("model")->as_string() : cfg.model_name;

        auto prompt_ids = tok.encode(prompt, true);
        if ((int)prompt_ids.size() >= cfg.max_ctx) {
            send_error(fd, 400, "Prompt exceeds max_ctx");
            return;
        }
        if ((int)prompt_ids.size() + max_tokens > cfg.max_ctx) {
            max_tokens = cfg.max_ctx - (int)prompt_ids.size();
        }

        std::string req_id = "cmpl-" + random_hex(24);
        int64_t created = (int64_t)std::time(nullptr);
        std::string pending, full_out;

        if (!stream) {
            bool hit_stop = false;
            GenStats stats;
            {
                std::lock_guard<std::mutex> lk(gpu_mu);
                stats = engine.generate(prompt_ids, max_tokens, req_mtp, [&](int32_t tid) -> bool {
                    if (tid == tok.id_eos) return false;
                    pending += tok.decode_token(tid, true);
                    size_t ok = valid_utf8_prefix_len(pending);
                    if (ok > 0) {
                        full_out.append(pending.data(), ok);
                        pending.erase(0, ok);
                    }
                    if (!stops.empty()) {
                        size_t idx = find_first_stop(full_out, stops);
                        if (idx != std::string::npos) {
                            full_out.resize(idx);
                            hit_stop = true;
                            return false;
                        }
                    }
                    return true;
                });
            }
            bool hit_eos = (!stats.tokens.empty() && stats.tokens.back() == EOS_TOKEN);
            double dec_toks = std::max(0, stats.gen_tokens - 1);
            double tps = (stats.decode_ms > 0.0) ? (dec_toks * 1000.0 / stats.decode_ms) : 0.0;

            Json ch = Json::object();
            ch["index"] = 0;
            ch["text"] = echo ? (prompt + full_out) : full_out;
            ch["logprobs"] = nullptr;
            ch["finish_reason"] = (hit_stop || hit_eos) ? "stop" : "length";

            Json resp = Json::object();
            resp["id"] = req_id;
            resp["object"] = "text_completion";
            resp["created"] = created;
            resp["model"] = req_model;
            resp["choices"] = Json::array();
            resp["choices"].push_back(std::move(ch));
            resp["usage"] = Json::object();
            resp["usage"]["prompt_tokens"] = stats.prompt_tokens;
            resp["usage"]["completion_tokens"] = stats.gen_tokens;
            resp["usage"]["total_tokens"] = stats.prompt_tokens + stats.gen_tokens;
            resp["timings"] = Json::object();
            resp["timings"]["prefill_ms"] = stats.prefill_ms;
            resp["timings"]["decode_ms"] = stats.decode_ms;
            resp["timings"]["tokens_per_second"] = tps;
            send_json(fd, 200, resp);
            return;
        }

        if (!send_sse_headers(fd)) return;
        auto send_cmpl = [&](const std::string& text, const Json& finish) -> bool {
            Json ch = Json::object();
            ch["index"] = 0;
            ch["text"] = text;
            ch["logprobs"] = nullptr;
            ch["finish_reason"] = finish;
            Json c = Json::object();
            c["id"] = req_id;
            c["object"] = "text_completion";
            c["created"] = created;
            c["model"] = req_model;
            c["choices"] = Json::array();
            c["choices"].push_back(std::move(ch));
            return send_sse_json(fd, c);
        };

        bool hit_stop = false;
        bool conn_ok = true;
        size_t emitted_len = 0;
        GenStats stats;
        {
            std::lock_guard<std::mutex> lk(gpu_mu);
            stats = engine.generate(prompt_ids, max_tokens, req_mtp, [&](int32_t tid) -> bool {
                if (tid == tok.id_eos) return false;
                pending += tok.decode_token(tid, true);
                size_t ok = valid_utf8_prefix_len(pending);
                if (ok == 0) return true;
                std::string piece(pending.data(), ok);
                pending.erase(0, ok);
                full_out += piece;
                if (!stops.empty()) {
                    size_t idx = find_first_stop(full_out, stops);
                    if (idx != std::string::npos) {
                        if (idx > emitted_len) send_cmpl(full_out.substr(emitted_len, idx - emitted_len), Json(nullptr));
                        emitted_len = idx;
                        hit_stop = true;
                        return false;
                    }
                    size_t hold = (max_stop_len > 1) ? (max_stop_len - 1) : 0;
                    if (full_out.size() > emitted_len + hold) {
                        size_t send_end = full_out.size() - hold;
                        if (!send_cmpl(full_out.substr(emitted_len, send_end - emitted_len), Json(nullptr))) {
                            conn_ok = false;
                            return false;
                        }
                        emitted_len = send_end;
                    }
                } else {
                    if (!send_cmpl(piece, Json(nullptr))) {
                        conn_ok = false;
                        return false;
                    }
                    emitted_len = full_out.size();
                }
                return true;
            });
        }
        if (!conn_ok) return;
        if (!stops.empty() && !hit_stop && emitted_len < full_out.size()) {
            send_cmpl(full_out.substr(emitted_len), Json(nullptr));
        }
        bool hit_eos = (!stats.tokens.empty() && stats.tokens.back() == EOS_TOKEN);
        send_cmpl("", Json((hit_stop || hit_eos) ? "stop" : "length"));
        send_sse_done(fd);
    }
};

} // namespace

int run_openai_server(Engine& engine, const Tokenizer& tokenizer, const ServerConfig& cfg) {
    ::signal(SIGPIPE, SIG_IGN);
    ::signal(SIGINT, handle_signal);
    ::signal(SIGTERM, handle_signal);
    g_stop_requested.store(false);

    // Warm up CUDA graphs once before accepting traffic
    {
        std::vector<int32_t> warm = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        (void)engine.generate(warm, 8, cfg.default_mtp);
    }

    int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        std::perror("socket");
        return 1;
    }
    int one = 1;
    ::setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)cfg.port);
    if (cfg.host == "0.0.0.0" || cfg.host.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (::inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "[server] Invalid bind host: %s\n", cfg.host.c_str());
        ::close(sfd);
        return 1;
    }

    if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind");
        ::close(sfd);
        return 1;
    }
    if (::listen(sfd, 64) < 0) {
        std::perror("listen");
        ::close(sfd);
        return 1;
    }
    g_listen_fd.store(sfd);

    printf("[server] Native C++/CUDA OpenAI API ready on http://%s:%d/v1 | model=%s | ctx=%d | mtp=%d\n",
           cfg.host.c_str(), cfg.port, cfg.model_name.c_str(), cfg.max_ctx, cfg.default_mtp);
    fflush(stdout);

    ServerContext ctx{engine, tokenizer, cfg, {}};

    while (!g_stop_requested.load()) {
        sockaddr_in caddr{};
        socklen_t clen = sizeof(caddr);
        int cfd = ::accept(sfd, reinterpret_cast<sockaddr*>(&caddr), &clen);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            if (g_stop_requested.load()) break;
            continue;
        }
        std::thread([&ctx, cfd]() { ctx.handle_client(cfd); }).detach();
    }

    int fd = g_listen_fd.exchange(-1);
    if (fd >= 0) ::close(fd);
    printf("\n[server] Shutdown complete.\n");
    return 0;
}

} // namespace squid
