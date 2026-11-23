#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace squid {

struct Json {
    enum class Type : uint8_t { Null, Bool, Int, Double, String, Array, Object };

    Type type = Type::Null;
    bool bval = false;
    int64_t ival = 0;
    double dval = 0.0;
    std::string sval;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    Json() = default;
    Json(std::nullptr_t) : type(Type::Null) {}
    Json(bool b) : type(Type::Bool), bval(b) {}
    Json(int v) : type(Type::Int), ival(v), dval(v) {}
    Json(int64_t v) : type(Type::Int), ival(v), dval((double)v) {}
    Json(size_t v) : type(Type::Int), ival((int64_t)v), dval((double)v) {}
    Json(double v) : type(Type::Double), dval(v) {}
    Json(const char* s) : type(Type::String), sval(s ? s : "") {}
    Json(std::string s) : type(Type::String), sval(std::move(s)) {}

    static Json array() { Json j; j.type = Type::Array; return j; }
    static Json object() { Json j; j.type = Type::Object; return j; }

    bool is_null()   const { return type == Type::Null; }
    bool is_bool()   const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Int || type == Type::Double; }
    bool is_string() const { return type == Type::String; }
    bool is_array()  const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool as_bool(bool def = false) const { return type == Type::Bool ? bval : def; }
    int64_t as_int(int64_t def = 0) const {
        if (type == Type::Int) return ival;
        if (type == Type::Double) return (int64_t)dval;
        return def;
    }
    double as_double(double def = 0.0) const {
        if (type == Type::Double) return dval;
        if (type == Type::Int) return (double)ival;
        return def;
    }
    const std::string& as_string() const { return sval; }

    bool contains(const std::string& k) const {
        return type == Type::Object && obj.find(k) != obj.end();
    }
    const Json* get(const std::string& k) const {
        if (type != Type::Object) return nullptr;
        auto it = obj.find(k);
        return (it != obj.end()) ? &it->second : nullptr;
    }
    Json& operator[](const std::string& k) {
        if (type != Type::Object) { type = Type::Object; obj.clear(); }
        return obj[k];
    }
    void push_back(Json v) {
        if (type != Type::Array) { type = Type::Array; arr.clear(); }
        arr.push_back(std::move(v));
    }

    static void append_escaped(std::string& out, std::string_view s) {
        out.push_back('"');
        for (unsigned char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out.push_back((char)c);
                    }
            }
        }
        out.push_back('"');
    }

    void dump_to(std::string& out) const {
        switch (type) {
            case Type::Null: out += "null"; break;
            case Type::Bool: out += (bval ? "true" : "false"); break;
            case Type::Int:  out += std::to_string(ival); break;
            case Type::Double: {
                if (!std::isfinite(dval)) { out += "null"; break; }
                char buf[48];
                std::snprintf(buf, sizeof(buf), "%.6g", dval);
                out += buf;
                break;
            }
            case Type::String: append_escaped(out, sval); break;
            case Type::Array:
                out.push_back('[');
                for (size_t i = 0; i < arr.size(); ++i) {
                    if (i) out.push_back(',');
                    arr[i].dump_to(out);
                }
                out.push_back(']');
                break;
            case Type::Object: {
                out.push_back('{');
                bool first = true;
                for (const auto& kv : obj) {
                    if (!first) out.push_back(',');
                    first = false;
                    append_escaped(out, kv.first);
                    out.push_back(':');
                    kv.second.dump_to(out);
                }
                out.push_back('}');
                break;
            }
        }
    }

    std::string dump() const {
        std::string out;
        out.reserve(128);
        dump_to(out);
        return out;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view src) : s_(src) {}

    Json parse() {
        skip_ws();
        Json v = parse_value();
        skip_ws();
        return v;
    }

    // Fast utility: parse a JSON string literal starting at `p` (must point to '"')
    static bool parse_string_at(const char* data, size_t len, size_t& pos, std::string& out) {
        if (pos >= len || data[pos] != '"') return false;
        ++pos;
        out.clear();
        while (pos < len) {
            unsigned char c = (unsigned char)data[pos++];
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back((char)c);
                continue;
            }
            if (pos >= len) return false;
            char esc = data[pos++];
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    if (pos + 4 > len) return false;
                    uint32_t cp = parse_hex4(data + pos);
                    pos += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= len &&
                        data[pos] == '\\' && data[pos + 1] == 'u') {
                        uint32_t lo = parse_hex4(data + pos + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            pos += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

private:
    std::string_view s_;
    size_t p_ = 0;

    static uint32_t parse_hex4(const char* h) {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = h[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else throw std::runtime_error("Invalid hex escape in JSON");
        }
        return v;
    }

    void skip_ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_;
    }

    Json parse_value() {
        skip_ws();
        if (p_ >= s_.size()) throw std::runtime_error("Unexpected end of JSON");
        char c = s_[p_];
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't') { expect("true"); return Json(true); }
        if (c == 'f') { expect("false"); return Json(false); }
        if (c == 'n') { expect("null"); return Json(nullptr); }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw std::runtime_error(std::string("Unexpected JSON char: ") + c);
    }

    void expect(const char* lit) {
        for (; *lit; ++lit, ++p_) {
            if (p_ >= s_.size() || s_[p_] != *lit) throw std::runtime_error("Invalid JSON literal");
        }
    }

    Json parse_string() {
        std::string out;
        if (!parse_string_at(s_.data(), s_.size(), p_, out)) {
            throw std::runtime_error("Unterminated or invalid JSON string");
        }
        return Json(std::move(out));
    }

    Json parse_number() {
        size_t start = p_;
        bool is_float = false;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        if (p_ < s_.size() && s_[p_] == '.') {
            is_float = true;
            ++p_;
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        }
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
            is_float = true;
            ++p_;
            if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        }
        std::string num(s_.substr(start, p_ - start));
        if (is_float) return Json(std::strtod(num.c_str(), nullptr));
        return Json((int64_t)std::strtoll(num.c_str(), nullptr, 10));
    }

    Json parse_array() {
        ++p_; // '['
        Json j = Json::array();
        skip_ws();
        if (p_ < s_.size() && s_[p_] == ']') { ++p_; return j; }
        while (true) {
            j.push_back(parse_value());
            skip_ws();
            if (p_ >= s_.size()) throw std::runtime_error("Unterminated JSON array");
            if (s_[p_] == ']') { ++p_; return j; }
            if (s_[p_] != ',') throw std::runtime_error("Expected ',' in JSON array");
            ++p_;
        }
    }

    Json parse_object() {
        ++p_; // '{'
        Json j = Json::object();
        skip_ws();
        if (p_ < s_.size() && s_[p_] == '}') { ++p_; return j; }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string_at(s_.data(), s_.size(), p_, key)) {
                throw std::runtime_error("Expected string key in JSON object");
            }
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != ':') throw std::runtime_error("Expected ':' in JSON object");
            ++p_;
            j[key] = parse_value();
            skip_ws();
            if (p_ >= s_.size()) throw std::runtime_error("Unterminated JSON object");
            if (s_[p_] == '}') { ++p_; return j; }
            if (s_[p_] != ',') throw std::runtime_error("Expected ',' in JSON object");
            ++p_;
        }
    }
};

inline Json json_parse(std::string_view s) {
    return JsonParser(s).parse();
}

} // namespace squid
