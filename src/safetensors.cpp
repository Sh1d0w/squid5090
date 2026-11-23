#include "safetensors.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace squid {

namespace {

// Minimal, zero-dependency parser for safetensors JSON header.
struct JsonParser {
    const char* s;
    size_t pos = 0;
    size_t len;

    void skip_ws() {
        while (pos < len && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t'))
            ++pos;
    }

    char peek() {
        skip_ws();
        return pos < len ? s[pos] : '\0';
    }

    void expect(char c) {
        skip_ws();
        if (pos >= len || s[pos] != c) {
            fprintf(stderr, "JSON parse error: expected '%c' at pos %zu\n", c, pos);
            std::exit(1);
        }
        ++pos;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos < len && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < len) {
                ++pos;
                out.push_back(s[pos++]);
            } else {
                out.push_back(s[pos++]);
            }
        }
        expect('"');
        return out;
    }

    int64_t parse_int() {
        skip_ws();
        char* end = nullptr;
        long long v = std::strtoll(s + pos, &end, 10);
        pos = static_cast<size_t>(end - s);
        return static_cast<int64_t>(v);
    }

    void skip_value() {
        char c = peek();
        if (c == '"') {
            (void)parse_string();
        } else if (c == '{') {
            expect('{');
            if (peek() != '}') {
                while (true) {
                    (void)parse_string();
                    expect(':');
                    skip_value();
                    if (peek() == ',') {
                        expect(',');
                    } else {
                        break;
                    }
                }
            }
            expect('}');
        } else if (c == '[') {
            expect('[');
            if (peek() != ']') {
                while (true) {
                    skip_value();
                    if (peek() == ',') {
                        expect(',');
                    } else {
                        break;
                    }
                }
            }
            expect(']');
        } else {
            while (pos < len && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' &&
                   s[pos] != ' ' && s[pos] != '\n' && s[pos] != '\r' && s[pos] != '\t')
                ++pos;
        }
    }
};

} // namespace

SafetensorsFile::~SafetensorsFile() { close(); }

void SafetensorsFile::close() {
    if (mapped_ && mapped_ != MAP_FAILED) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    tensors_.clear();
}

bool SafetensorsFile::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        perror(("open " + path).c_str());
        return false;
    }
    struct stat st{};
    if (fstat(fd_, &st) != 0) {
        perror("fstat");
        close();
        return false;
    }
    file_size_ = static_cast<size_t>(st.st_size);
    mapped_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (mapped_ == MAP_FAILED) {
        perror("mmap");
        mapped_ = nullptr;
        close();
        return false;
    }
    madvise(mapped_, file_size_, MADV_SEQUENTIAL);

    uint64_t header_len = 0;
    std::memcpy(&header_len, mapped_, 8);
    payload_offset_ = 8 + header_len;

    JsonParser p{reinterpret_cast<const char*>(mapped_ + 8), 0, static_cast<size_t>(header_len)};
    p.expect('{');
    if (p.peek() != '}') {
        while (true) {
            std::string key = p.parse_string();
            p.expect(':');
            if (key == "__metadata__") {
                p.skip_value();
            } else {
                TensorInfo info;
                info.name = key;
                p.expect('{');
                uint64_t off0 = 0, off1 = 0;
                if (p.peek() != '}') {
                    while (true) {
                        std::string fk = p.parse_string();
                        p.expect(':');
                        if (fk == "dtype") {
                            info.dtype = p.parse_string();
                        } else if (fk == "shape") {
                            p.expect('[');
                            if (p.peek() != ']') {
                                while (true) {
                                    info.shape.push_back(p.parse_int());
                                    if (p.peek() == ',') p.expect(',');
                                    else break;
                                }
                            }
                            p.expect(']');
                        } else if (fk == "data_offsets") {
                            p.expect('[');
                            off0 = static_cast<uint64_t>(p.parse_int());
                            p.expect(',');
                            off1 = static_cast<uint64_t>(p.parse_int());
                            p.expect(']');
                        } else {
                            p.skip_value();
                        }
                        if (p.peek() == ',') p.expect(',');
                        else break;
                    }
                }
                p.expect('}');
                info.data_offset = off0;
                info.nbytes = off1 - off0;
                tensors_.emplace(info.name, std::move(info));
            }
            if (p.peek() == ',') p.expect(',');
            else break;
        }
    }
    p.expect('}');
    return true;
}

bool SafetensorsFile::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const TensorInfo& SafetensorsFile::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        fprintf(stderr, "Missing tensor in safetensors: %s\n", name.c_str());
        std::exit(1);
    }
    return it->second;
}

const uint8_t* SafetensorsFile::data_ptr(const TensorInfo& info) const {
    return mapped_ + payload_offset_ + info.data_offset;
}

void SafetensorsFile::read_tensor(const std::string& name, void* dst) const {
    const TensorInfo& info = get(name);
    uint64_t file_off = payload_offset_ + info.data_offset;
    uint64_t remaining = info.nbytes;
    uint8_t* out = static_cast<uint8_t*>(dst);
    while (remaining > 0) {
        ssize_t n = ::pread(fd_, out, remaining, static_cast<off_t>(file_off));
        if (n <= 0) {
            perror(("pread " + name).c_str());
            std::exit(1);
        }
        out += n;
        file_off += static_cast<uint64_t>(n);
        remaining -= static_cast<uint64_t>(n);
    }
}

} // namespace squid
