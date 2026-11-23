#include "tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace squid {

namespace {

// Standard GPT-2 ByteLevel byte <-> Unicode codepoint table
struct ByteLevelTable {
    uint32_t byte_to_cp[256];
    int16_t cp_to_byte[512];

    ByteLevelTable() {
        for (int i = 0; i < 512; ++i) cp_to_byte[i] = -1;
        bool direct[256] = {};
        for (int b = 33; b <= 126; ++b) direct[b] = true;
        for (int b = 161; b <= 172; ++b) direct[b] = true;
        for (int b = 174; b <= 255; ++b) direct[b] = true;
        uint32_t next_cp = 256;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = direct[b] ? (uint32_t)b : next_cp++;
            byte_to_cp[b] = cp;
            cp_to_byte[cp] = (int16_t)b;
        }
    }
};

static const ByteLevelTable& byte_level_table() {
    static const ByteLevelTable tbl;
    return tbl;
}

// Decode a UTF-8 codepoint starting at s[pos], advancing pos
inline uint32_t next_utf8(const char* s, size_t len, size_t& pos) {
    unsigned char c0 = (unsigned char)s[pos++];
    if (c0 < 0x80) return c0;
    if ((c0 & 0xE0) == 0xC0 && pos < len) {
        uint32_t cp = ((c0 & 0x1Fu) << 6) | ((unsigned char)s[pos++] & 0x3Fu);
        return cp;
    }
    if ((c0 & 0xF0) == 0xE0 && pos + 1 < len) {
        uint32_t cp = ((c0 & 0x0Fu) << 12)
                    | (((unsigned char)s[pos] & 0x3Fu) << 6)
                    | ((unsigned char)s[pos + 1] & 0x3Fu);
        pos += 2;
        return cp;
    }
    if ((c0 & 0xF8) == 0xF0 && pos + 2 < len) {
        uint32_t cp = ((c0 & 0x07u) << 18)
                    | (((unsigned char)s[pos] & 0x3Fu) << 12)
                    | (((unsigned char)s[pos + 1] & 0x3Fu) << 6)
                    | ((unsigned char)s[pos + 2] & 0x3Fu);
        pos += 3;
        return cp;
    }
    return c0;
}

// Convert GPT-2 ByteLevel-encoded UTF-8 string into raw bytes
std::string decode_byte_level_str(std::string_view s) {
    const auto& tbl = byte_level_table();
    std::string out;
    out.reserve(s.size());
    size_t p = 0;
    while (p < s.size()) {
        size_t prev = p;
        uint32_t cp = next_utf8(s.data(), s.size(), p);
        if (cp < 512 && tbl.cp_to_byte[cp] >= 0) {
            out.push_back((char)tbl.cp_to_byte[cp]);
        } else {
            out.append(s.data() + prev, p - prev);
        }
    }
    return out;
}

// Fast Unicode character classification for o200k_base pre-tokenizer
inline bool cp_is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
inline bool cp_is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f' ||
           cp == 0x0085 || cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
inline bool cp_is_digit(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) || (cp >= 0xFF10 && cp <= 0xFF19);
}
inline bool cp_is_upper(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x00C0 && cp <= 0x00D6) || (cp >= 0x00D8 && cp <= 0x00DE) ||
           (cp >= 0x0391 && cp <= 0x03A9) || (cp >= 0x0410 && cp <= 0x042F) ||
           (cp >= 0xFF21 && cp <= 0xFF3A);
}
inline bool cp_is_lower(uint32_t cp) {
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp < 0x80) return false;
    return (cp >= 0x00DF && cp <= 0x00F6) || (cp >= 0x00F8 && cp <= 0x00FF) ||
           (cp >= 0x03B1 && cp <= 0x03C9) || (cp >= 0x0430 && cp <= 0x044F) ||
           (cp >= 0xFF41 && cp <= 0xFF5A);
}
inline bool cp_is_other_letter_or_mark(uint32_t cp) {
    if (cp < 0x80) return false;
    // Latin Extended, Combining Diacritics, Hebrew, Arabic, Devanagari, CJK, Kana, Hangul
    return (cp >= 0x0100 && cp <= 0x024F) || (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x0590 && cp <= 0x06FF) || (cp >= 0x0900 && cp <= 0x097F) ||
           (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x9FFF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF);
}
inline bool cp_is_letter(uint32_t cp) {
    return cp_is_upper(cp) || cp_is_lower(cp) || cp_is_other_letter_or_mark(cp);
}
inline bool cp_is_cat_U(uint32_t cp) { return cp_is_upper(cp) || cp_is_other_letter_or_mark(cp); }
inline bool cp_is_cat_L(uint32_t cp) { return cp_is_lower(cp) || cp_is_other_letter_or_mark(cp); }

// Match case-insensitive English contraction (?i:'s|'t|'re|'ve|'m|'ll|'d)
size_t match_contraction(const std::vector<uint32_t>& cps, size_t i) {
    size_t n = cps.size();
    if (i + 1 >= n || cps[i] != '\'') return 0;
    auto lo = [](uint32_t c) -> uint32_t { return (c >= 'A' && c <= 'Z') ? (c + 32) : c; };
    uint32_t c1 = lo(cps[i + 1]);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (i + 2 < n) {
        uint32_t c2 = lo(cps[i + 2]);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return 3;
    }
    return 0;
}

// Fast JSON scanner helpers for loading tokenizer.json without building a giant DOM
inline void skip_ws(const char* d, size_t n, size_t& p) {
    while (p < n && (d[p] == ' ' || d[p] == '\t' || d[p] == '\n' || d[p] == '\r')) ++p;
}

void skip_json_value(const char* d, size_t n, size_t& p) {
    skip_ws(d, n, p);
    if (p >= n) return;
    char c = d[p];
    if (c == '"') {
        std::string tmp;
        JsonParser::parse_string_at(d, n, p, tmp);
        return;
    }
    if (c == '{' || c == '[') {
        char close = (c == '{') ? '}' : ']';
        int depth = 1;
        ++p;
        while (p < n && depth > 0) {
            if (d[p] == '"') {
                ++p;
                while (p < n) {
                    if (d[p] == '\\') { p += 2; continue; }
                    if (d[p++] == '"') break;
                }
            } else {
                if (d[p] == c) ++depth;
                else if (d[p] == close) --depth;
                ++p;
            }
        }
        return;
    }
    while (p < n && d[p] != ',' && d[p] != '}' && d[p] != ']' &&
           d[p] != ' ' && d[p] != '\t' && d[p] != '\n' && d[p] != '\r') ++p;
}

int64_t parse_int_at(const char* d, size_t n, size_t& p) {
    skip_ws(d, n, p);
    bool neg = false;
    if (p < n && d[p] == '-') { neg = true; ++p; }
    int64_t v = 0;
    while (p < n && d[p] >= '0' && d[p] <= '9') {
        v = v * 10 + (d[p++] - '0');
    }
    return neg ? -v : v;
}

} // namespace

bool Tokenizer::load(const std::string& model_dir) {
    auto t0 = std::chrono::steady_clock::now();
    std::string path = model_dir + "/tokenizer.json";
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        fprintf(stderr, "[tokenizer] Failed to open %s\n", path.c_str());
        return false;
    }
    size_t sz = (size_t)ifs.tellg();
    ifs.seekg(0);
    std::string buf(sz, '\0');
    if (!ifs.read(buf.data(), (std::streamsize)sz)) return false;

    const char* d = buf.data();
    size_t n = buf.size();
    size_t p = 0;

    id_to_bytes_.clear();
    is_special_.clear();
    bytes_to_id_.clear();
    special_to_id_.clear();
    special_sorted_.clear();
    merges_.clear();
    bytes_to_id_.reserve(210000);
    merges_.reserve(450000);

    auto ensure_id = [&](int32_t id) {
        if ((size_t)id >= id_to_bytes_.size()) {
            id_to_bytes_.resize((size_t)id + 1);
            is_special_.resize((size_t)id + 1, 0);
        }
    };

    // Pass 1: parse top-level keys ("added_tokens", "model" -> {"vocab", "merges"})
    // Keep raw slices of merges so we parse vocab before merges regardless of JSON key order!
    size_t merges_pos = std::string::npos;

    skip_ws(d, n, p);
    if (p >= n || d[p] != '{') return false;
    ++p;
    while (p < n) {
        skip_ws(d, n, p);
        if (p < n && d[p] == '}') { ++p; break; }
        std::string key;
        if (!JsonParser::parse_string_at(d, n, p, key)) return false;
        skip_ws(d, n, p);
        if (p >= n || d[p] != ':') return false;
        ++p;
        skip_ws(d, n, p);

        if (key == "added_tokens" && p < n && d[p] == '[') {
            ++p;
            while (p < n) {
                skip_ws(d, n, p);
                if (p < n && d[p] == ']') { ++p; break; }
                if (d[p] == '{') {
                    ++p;
                    int32_t aid = -1;
                    std::string content;
                    while (p < n) {
                        skip_ws(d, n, p);
                        if (p < n && d[p] == '}') { ++p; break; }
                        std::string ak;
                        JsonParser::parse_string_at(d, n, p, ak);
                        skip_ws(d, n, p);
                        if (p < n && d[p] == ':') ++p;
                        skip_ws(d, n, p);
                        if (ak == "id") aid = (int32_t)parse_int_at(d, n, p);
                        else if (ak == "content") JsonParser::parse_string_at(d, n, p, content);
                        else skip_json_value(d, n, p);
                        skip_ws(d, n, p);
                        if (p < n && d[p] == ',') ++p;
                    }
                    if (aid >= 0 && !content.empty()) {
                        ensure_id(aid);
                        id_to_bytes_[aid] = content;
                        is_special_[aid] = 1;
                        special_to_id_[content] = aid;
                        special_sorted_.push_back({content, aid});
                    }
                } else {
                    skip_json_value(d, n, p);
                }
                skip_ws(d, n, p);
                if (p < n && d[p] == ',') ++p;
            }
        } else if (key == "model" && p < n && d[p] == '{') {
            ++p;
            while (p < n) {
                skip_ws(d, n, p);
                if (p < n && d[p] == '}') { ++p; break; }
                std::string mk;
                if (!JsonParser::parse_string_at(d, n, p, mk)) return false;
                skip_ws(d, n, p);
                if (p >= n || d[p] != ':') return false;
                ++p;
                skip_ws(d, n, p);
                if (mk == "vocab" && p < n && d[p] == '{') {
                    ++p;
                    std::string tok_str;
                    while (p < n) {
                        skip_ws(d, n, p);
                        if (p < n && d[p] == '}') { ++p; break; }
                        if (!JsonParser::parse_string_at(d, n, p, tok_str)) return false;
                        skip_ws(d, n, p);
                        if (p < n && d[p] == ':') ++p;
                        int32_t id = (int32_t)parse_int_at(d, n, p);
                        std::string raw = decode_byte_level_str(tok_str);
                        ensure_id(id);
                        id_to_bytes_[id] = raw;
                        bytes_to_id_[raw] = id;
                        skip_ws(d, n, p);
                        if (p < n && d[p] == ',') ++p;
                    }
                } else if (mk == "merges" && p < n && d[p] == '[') {
                    merges_pos = p;
                    skip_json_value(d, n, p);
                } else {
                    skip_json_value(d, n, p);
                }
                skip_ws(d, n, p);
                if (p < n && d[p] == ',') ++p;
            }
        } else {
            skip_json_value(d, n, p);
        }
        skip_ws(d, n, p);
        if (p < n && d[p] == ',') ++p;
    }

    // Parse merges array now that vocab is loaded
    if (merges_pos != std::string::npos) {
        size_t mp = merges_pos + 1;
        uint32_t rank = 0;
        std::string a_str, b_str;
        while (mp < n) {
            skip_ws(d, n, mp);
            if (mp < n && d[mp] == ']') break;
            if (d[mp] == '[') {
                ++mp;
                skip_ws(d, n, mp);
                JsonParser::parse_string_at(d, n, mp, a_str);
                skip_ws(d, n, mp);
                if (mp < n && d[mp] == ',') ++mp;
                skip_ws(d, n, mp);
                JsonParser::parse_string_at(d, n, mp, b_str);
                skip_ws(d, n, mp);
                if (mp < n && d[mp] == ']') ++mp;
            } else if (d[mp] == '"') {
                std::string pair;
                JsonParser::parse_string_at(d, n, mp, pair);
                size_t sp = pair.find(' ');
                if (sp == std::string::npos) { ++rank; continue; }
                a_str = pair.substr(0, sp);
                b_str = pair.substr(sp + 1);
            } else {
                skip_json_value(d, n, mp);
                continue;
            }
            std::string ra = decode_byte_level_str(a_str);
            std::string rb = decode_byte_level_str(b_str);
            auto ita = bytes_to_id_.find(ra);
            auto itb = bytes_to_id_.find(rb);
            if (ita != bytes_to_id_.end() && itb != bytes_to_id_.end()) {
                std::string rab = ra + rb;
                auto itab = bytes_to_id_.find(rab);
                if (itab != bytes_to_id_.end()) {
                    uint64_t key = ((uint64_t)(uint32_t)ita->second << 32) | (uint32_t)itb->second;
                    merges_.emplace(key, MergeEntry{rank, itab->second});
                }
            }
            ++rank;
            skip_ws(d, n, mp);
            if (mp < n && d[mp] == ',') ++mp;
        }
    }

    // Build single-byte lookup
    for (int b = 0; b < 256; ++b) {
        std::string s(1, (char)b);
        auto it = bytes_to_id_.find(s);
        byte_to_id_[b] = (it != bytes_to_id_.end()) ? it->second : 0;
    }

    // Sort special tokens longest-first for greedy matching
    std::sort(special_sorted_.begin(), special_sorted_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    auto lookup = [&](const char* name, int32_t fallback) -> int32_t {
        auto it = special_to_id_.find(name);
        return (it != special_to_id_.end()) ? it->second : fallback;
    };
    id_msg_model = lookup("<|message_model|>", 200001);
    id_thinking  = lookup("<|content_thinking|>", 200008);
    id_text      = lookup("<|content_text|>", 200004);
    id_tool_json = lookup("<|content_invoke_tool_json|>", -1);
    id_end_msg   = lookup("<|end_message|>", 200010);
    id_eos       = lookup("<|content_model_end_sampling|>", 200006);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[tokenizer] Loaded %zu vocab + %zu merges + %zu special tokens in %.1fms\n",
           bytes_to_id_.size(), merges_.size(), special_sorted_.size(), ms);
    return !bytes_to_id_.empty();
}

int32_t Tokenizer::token_to_id(const std::string& tok) const {
    auto its = special_to_id_.find(tok);
    if (its != special_to_id_.end()) return its->second;
    auto it = bytes_to_id_.find(tok);
    return (it != bytes_to_id_.end()) ? it->second : -1;
}

void Tokenizer::encode_bpe_piece(std::string_view piece, std::vector<int32_t>& out) const {
    if (piece.empty()) return;
    if (piece.size() == 1) {
        out.push_back(byte_to_id_[(unsigned char)piece[0]]);
        return;
    }
    std::vector<int32_t> ids;
    ids.reserve(piece.size());
    for (unsigned char b : piece) ids.push_back(byte_to_id_[b]);

    while (ids.size() >= 2) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        int32_t best_merged = -1;
        size_t best_idx = 0;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            uint64_t key = ((uint64_t)(uint32_t)ids[i] << 32) | (uint32_t)ids[i + 1];
            auto it = merges_.find(key);
            if (it != merges_.end() && it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best_merged = it->second.merged_id;
                best_idx = i;
            }
        }
        if (best_rank == std::numeric_limits<uint32_t>::max()) break;
        // Merge ALL non-overlapping occurrences of the winning pair (identical to HF BPE!)
        int32_t a = ids[best_idx];
        int32_t b = ids[best_idx + 1];
        size_t w = 0;
        for (size_t r = 0; r < ids.size(); ) {
            if (r + 1 < ids.size() && ids[r] == a && ids[r + 1] == b) {
                ids[w++] = best_merged;
                r += 2;
            } else {
                ids[w++] = ids[r++];
            }
        }
        ids.resize(w);
    }
    out.insert(out.end(), ids.begin(), ids.end());
}

void Tokenizer::pretokenize_and_encode(std::string_view text, std::vector<int32_t>& out) const {
    if (text.empty()) return;
    // Decode UTF-8 into parallel codepoints + byte offsets
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    cps.reserve(text.size());
    offs.reserve(text.size() + 1);
    size_t bp = 0;
    while (bp < text.size()) {
        offs.push_back(bp);
        cps.push_back(next_utf8(text.data(), text.size(), bp));
    }
    offs.push_back(text.size());

    size_t n = cps.size();
    size_t i = 0;
    while (i < n) {
        size_t end = i;

        // Branch 1 & 2: optional [^\r\n\p{L}\p{N}]? followed by letter sequence + optional contraction
        auto try_branch1 = [&](size_t start) -> size_t {
            size_t j = start;
            if (j < n && !cp_is_newline(cps[j]) && !cp_is_letter(cps[j]) && !cp_is_digit(cps[j])) {
                ++j;
            }
            // Need cat_U* followed by cat_L+ (greedy cat_U*, then cat_L+)
            size_t run_start = j;
            while (j < n && cp_is_letter(cps[j])) ++j;
            // Check if there's any cat_L in [run_start..j): find the LAST index where all preceding are cat_U
            // Actually cat_U* cat_L+ matches iff there is a split k in [run_start..j-1] with [run_start..k) in cat_U and [k..j) in cat_L
            // Since cat_U = upper|other and cat_L = lower|other:
            // Scan from run_start: while cat_U (preferring upper before lower)
            size_t u_end = run_start;
            while (u_end < j && cp_is_cat_U(cps[u_end])) ++u_end;
            // First non-cat_U must be lower (since all chars in [run_start..j) are letters),
            // and all remaining chars of branch 1 must be cat_L!
            size_t l_start = u_end;
            if (l_start == j && u_end > run_start && cp_is_cat_L(cps[u_end - 1])) {
                // Backtrack 1 char if the last cat_U char is also cat_L (e.g. CJK/other)
                l_start = u_end - 1;
            }
            size_t l_end = l_start;
            while (l_end < j && cp_is_cat_L(cps[l_end])) ++l_end;
            if (l_end > l_start) {
                l_end += match_contraction(cps, l_end);
                return l_end;
            }
            return 0;
        };

        auto try_branch2 = [&](size_t start) -> size_t {
            size_t j = start;
            if (j < n && !cp_is_newline(cps[j]) && !cp_is_letter(cps[j]) && !cp_is_digit(cps[j])) {
                ++j;
            }
            size_t u_start = j;
            while (j < n && cp_is_cat_U(cps[j])) ++j;
            if (j == u_start) return 0;
            while (j < n && cp_is_cat_L(cps[j])) ++j;
            j += match_contraction(cps, j);
            return j;
        };

        size_t m1 = try_branch1(i);
        if (m1 > i) {
            end = m1;
        } else {
            size_t m2 = try_branch2(i);
            if (m2 > i) {
                end = m2;
            } else if (cp_is_digit(cps[i])) {
                // Branch 3: \p{N}{1,3}
                end = i + 1;
                if (end < n && cp_is_digit(cps[end])) ++end;
                if (end < n && cp_is_digit(cps[end])) ++end;
            } else {
                // Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
                size_t j = i;
                if (cps[j] == ' ' && j + 1 < n && !cp_is_space(cps[j + 1]) && !cp_is_letter(cps[j + 1]) && !cp_is_digit(cps[j + 1])) {
                    ++j;
                }
                if (!cp_is_space(cps[j]) && !cp_is_letter(cps[j]) && !cp_is_digit(cps[j])) {
                    while (j < n && !cp_is_space(cps[j]) && !cp_is_letter(cps[j]) && !cp_is_digit(cps[j])) ++j;
                    while (j < n && (cps[j] == '\r' || cps[j] == '\n' || cps[j] == '/')) ++j;
                    end = j;
                } else if (cp_is_space(cps[i])) {
                    // Branches 5, 6, 7: whitespace runs
                    size_t ws_end = i;
                    size_t last_nl = std::string::npos;
                    while (ws_end < n && cp_is_space(cps[ws_end])) {
                        if (cp_is_newline(cps[ws_end])) last_nl = ws_end;
                        ++ws_end;
                    }
                    if (last_nl != std::string::npos) {
                        // Branch 5: \s*[\r\n]+
                        end = last_nl + 1;
                    } else if (ws_end == n) {
                        // Branch 6: \s+(?!\S) at end of string
                        end = ws_end;
                    } else if (ws_end - i > 1) {
                        // Branch 6: \s+(?!\S) leaves 1 space for the following token
                        end = ws_end - 1;
                    } else {
                        // Branch 7: single \s+
                        end = i + 1;
                    }
                } else {
                    end = i + 1;
                }
            }
        }

        encode_bpe_piece(text.substr(offs[i], offs[end] - offs[i]), out);
        i = end;
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<int32_t> out;
    out.reserve(text.size() / 3 + 16);
    if (!parse_special || special_sorted_.empty()) {
        pretokenize_and_encode(text, out);
        return out;
    }
    size_t pos = 0;
    size_t seg_start = 0;
    while (pos < text.size()) {
        // Fast skip: all Inkling special tokens start with "<|"
        size_t cand = text.find("<|", pos);
        if (cand == std::string_view::npos) break;
        int32_t matched_id = -1;
        size_t matched_len = 0;
        for (const auto& sp : special_sorted_) {
            if (cand + sp.first.size() <= text.size() &&
                std::memcmp(text.data() + cand, sp.first.data(), sp.first.size()) == 0) {
                matched_id = sp.second;
                matched_len = sp.first.size();
                break;
            }
        }
        if (matched_id >= 0) {
            if (cand > seg_start) {
                pretokenize_and_encode(text.substr(seg_start, cand - seg_start), out);
            }
            out.push_back(matched_id);
            pos = cand + matched_len;
            seg_start = pos;
        } else {
            pos = cand + 2;
        }
    }
    if (seg_start < text.size()) {
        pretokenize_and_encode(text.substr(seg_start), out);
    }
    return out;
}

std::string Tokenizer::decode_token(int32_t id, bool skip_special) const {
    if (id < 0 || (size_t)id >= id_to_bytes_.size()) return "";
    if (skip_special && is_special_[id]) return "";
    return id_to_bytes_[id];
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string out;
    out.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= id_to_bytes_.size()) continue;
        if (skip_special && is_special_[id]) continue;
        out += id_to_bytes_[id];
    }
    return out;
}

std::string Tokenizer::render_chat_template(
    const Json& messages,
    const Json* tools,
    const Json* reasoning_effort) const {
    // Exact native C++ implementation of Inkling-2-Mini's official chat_template.jinja!
    std::string out;
    out.reserve(1024);

    auto emit_effort = [&]() {
        double num = 0.0;
        if (reasoning_effort && !reasoning_effort->is_null()) {
            if (reasoning_effort->is_string()) {
                const std::string& k = reasoning_effort->as_string();
                if (k == "none") num = 0.0;
                else if (k == "minimal") num = 0.1;
                else if (k == "low") num = 0.2;
                else if (k == "medium") num = 0.7;
                else if (k == "high") num = 0.9;
                else if (k == "xhigh" || k == "max") num = 0.99;
                else num = std::strtod(k.c_str(), nullptr);
            } else if (reasoning_effort->is_number()) {
                num = reasoning_effort->as_double(0.0);
            }
        }
        if (num < 0.0) num = 0.0;
        if (num > 0.99) num = 0.99;
        out += "<|message_system|><|content_text|>Thinking effort level: ";
        if (num == 0.0) {
            out += "0";
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.6g", num);
            out += buf;
        }
        out += "<|end_message|>";
    };

    if (tools && tools->is_array() && !tools->arr.empty()) {
        Json specs = Json::array();
        for (const auto& t : tools->arr) {
            const Json* fn = t.get("function") ? t.get("function") : &t;
            Json spec = Json::object();
            spec["description"] = (fn->get("description") && fn->get("description")->is_string())
                                ? fn->get("description")->as_string() : "";
            spec["name"] = (fn->get("name") && fn->get("name")->is_string())
                         ? fn->get("name")->as_string() : "";
            spec["parameters"] = fn->get("parameters") ? *fn->get("parameters") : Json::object();
            spec["type"] = (t.get("type") && t.get("type")->is_string())
                         ? t.get("type")->as_string() : "function";
            specs.push_back(std::move(spec));
        }
        out += "<|message_system|>tool_declare<|content_xml|>";
        specs.dump_to(out);
        out += "<|end_message|>";
    }

    bool effort_emitted = false;
    if (messages.is_array()) {
        for (size_t mi = 0; mi < messages.arr.size(); ++mi) {
            const Json& m = messages.arr[mi];
            std::string role = (m.get("role") && m.get("role")->is_string()) ? m.get("role")->as_string() : "user";

            if (!effort_emitted && role != "system") {
                emit_effort();
                effort_emitted = true;
            }

            std::string rtok;
            if (role == "user") rtok = "<|message_user|>";
            else if (role == "assistant") rtok = "<|message_model|>";
            else if (role == "system") rtok = "<|message_system|>";
            else if (role == "tool") rtok = "<|message_tool|>";
            else rtok = "<|message_user|>";

            if (role == "tool") {
                if (m.get("name") && m.get("name")->is_string() && !m.get("name")->as_string().empty()) {
                    rtok += m.get("name")->as_string();
                } else if (m.get("tool_call_id") && m.get("tool_call_id")->is_string()) {
                    const std::string& tcid = m.get("tool_call_id")->as_string();
                    for (size_t pi = 0; pi < mi; ++pi) {
                        const Json& pm = messages.arr[pi];
                        if (pm.get("role") && pm.get("role")->as_string() == "assistant" &&
                            pm.get("tool_calls") && pm.get("tool_calls")->is_array()) {
                            for (const auto& tc : pm.get("tool_calls")->arr) {
                                if (tc.get("id") && tc.get("id")->as_string() == tcid &&
                                    tc.get("function") && tc.get("function")->get("name")) {
                                    rtok += tc.get("function")->get("name")->as_string();
                                }
                            }
                        }
                    }
                }
            } else if (role == "assistant" && m.get("name") && m.get("name")->is_string()) {
                rtok += m.get("name")->as_string();
            }

            bool has_reasoning = (role == "assistant" && m.get("reasoning_content") &&
                                  m.get("reasoning_content")->is_string() &&
                                  !m.get("reasoning_content")->as_string().empty());
            bool has_tool_calls = (role == "assistant" && m.get("tool_calls") &&
                                   m.get("tool_calls")->is_array() && !m.get("tool_calls")->arr.empty());

            if (has_reasoning) {
                out += rtok + "<|content_thinking|>" + m.get("reasoning_content")->as_string() + "<|end_message|>";
            }

            const Json* c = m.get("content");
            if (c && c->is_string()) {
                if (!c->as_string().empty() || !(role == "assistant" && (has_tool_calls || has_reasoning))) {
                    out += rtok + "<|content_text|>" + c->as_string() + "<|end_message|>";
                }
            } else if (c && c->is_array()) {
                for (const auto& part : c->arr) {
                    if (part.is_string()) {
                        out += rtok + "<|content_text|>" + part.as_string() + "<|end_message|>";
                    } else if (part.is_object()) {
                        std::string ptype = (part.get("type") && part.get("type")->is_string())
                                          ? part.get("type")->as_string() : "text";
                        if (ptype == "text" || ptype == "input_text") {
                            std::string tp = (part.get("text") && part.get("text")->is_string())
                                           ? part.get("text")->as_string() : "";
                            out += rtok + "<|content_text|>" + tp + "<|end_message|>";
                        }
                    }
                }
            }

            if (has_tool_calls) {
                for (const auto& tc : m.get("tool_calls")->arr) {
                    const Json* fn = tc.get("function");
                    if (!fn || !fn->get("name")) continue;
                    std::string fname = fn->get("name")->as_string();
                    Json args_obj = Json::object();
                    if (const Json* a = fn->get("arguments")) {
                        if (a->is_object()) args_obj = *a;
                        else if (a->is_string() && !a->as_string().empty()) {
                            try { args_obj = json_parse(a->as_string()); } catch (...) {}
                        }
                    }
                    out += "<|message_model|>" + fname + "<|content_invoke_tool_json|>{\"name\":";
                    Json::append_escaped(out, fname);
                    out += ",\"args\":";
                    args_obj.dump_to(out);
                    out += "}<|end_message|>";
                }
            }

            if (role == "assistant") {
                out += "<|content_model_end_sampling|>";
            }
        }
    }

    if (!effort_emitted) emit_effort();
    return out;
}

} // namespace squid
