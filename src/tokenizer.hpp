#pragma once

#include "json.hpp"
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace squid {

class Tokenizer {
public:
    bool load(const std::string& model_dir);

    // Encode UTF-8 text (preserving any special/added tokens like <|message_user|>)
    std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;

    // Decode token IDs back to UTF-8 string
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;
    std::string decode_token(int32_t id, bool skip_special = false) const;

    // Render OpenAI-format messages (+ optional tools & reasoning_effort) using Inkling-2-Mini's chat format
    std::string render_chat_template(
        const Json& messages,
        const Json* tools = nullptr,
        const Json* reasoning_effort = nullptr) const;

    int32_t token_to_id(const std::string& tok) const;

    // Inkling-2-Mini special token IDs (cached on load)
    int32_t id_msg_model = 200001;
    int32_t id_thinking  = 200008;
    int32_t id_text      = 200004;
    int32_t id_tool_json = -1;
    int32_t id_end_msg   = 200010;
    int32_t id_eos       = 200006;

private:
    struct MergeEntry {
        uint32_t rank;
        int32_t merged_id;
    };

    void encode_bpe_piece(std::string_view piece, std::vector<int32_t>& out) const;
    void pretokenize_and_encode(std::string_view text, std::vector<int32_t>& out) const;

    std::vector<std::string> id_to_bytes_;
    std::vector<uint8_t> is_special_;
    std::unordered_map<std::string, int32_t> bytes_to_id_;
    std::unordered_map<std::string, int32_t> special_to_id_;
    std::vector<std::pair<std::string, int32_t>> special_sorted_; // sorted longest-first
    std::unordered_map<uint64_t, MergeEntry> merges_;
    int32_t byte_to_id_[256] = {};
};

} // namespace squid
