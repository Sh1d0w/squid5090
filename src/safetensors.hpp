#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace squid {

struct TensorInfo {
    std::string name;
    std::string dtype; // "F32", "BF16", "U8", "F8_E4M3", etc.
    std::vector<int64_t> shape;
    uint64_t data_offset = 0; // relative to start of tensor payload
    uint64_t nbytes = 0;
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    bool open(const std::string& path);
    void close();

    const TensorInfo& get(const std::string& name) const;
    bool has(const std::string& name) const;
    const uint8_t* data_ptr(const TensorInfo& info) const;
    const uint8_t* data_ptr(const std::string& name) const { return data_ptr(get(name)); }

    // Eager pread into caller buffer (fast on NVMe / avoids page-fault stalls)
    void read_tensor(const std::string& name, void* dst) const;

    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }

private:
    int fd_ = -1;
    uint8_t* mapped_ = nullptr;
    size_t file_size_ = 0;
    uint64_t payload_offset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

} // namespace squid
