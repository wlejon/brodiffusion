#include "brodiffusion/safetensors.h"

#include "brotensor/tensor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace brodiffusion::safetensors {

// ─── Dtype helpers ─────────────────────────────────────────────────────────

const char* dtype_name(Dtype d) {
    switch (d) {
        case Dtype::F32:  return "F32";
        case Dtype::F16:  return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::I32:  return "I32";
        case Dtype::I64:  return "I64";
        case Dtype::U8:   return "U8";
        case Dtype::BOOL: return "BOOL";
        case Dtype::Unknown:
        default:          return "Unknown";
    }
}

int dtype_size_bytes(Dtype d) {
    switch (d) {
        case Dtype::F32:  return 4;
        case Dtype::F16:  return 2;
        case Dtype::BF16: return 2;
        case Dtype::I32:  return 4;
        case Dtype::I64:  return 8;
        case Dtype::U8:   return 1;
        case Dtype::BOOL: return 1;
        default:          return 0;
    }
}

static Dtype parse_dtype(std::string_view s) {
    if (s == "F32")  return Dtype::F32;
    if (s == "F16")  return Dtype::F16;
    if (s == "BF16") return Dtype::BF16;
    if (s == "I32")  return Dtype::I32;
    if (s == "I64")  return Dtype::I64;
    if (s == "U8")   return Dtype::U8;
    if (s == "BOOL") return Dtype::BOOL;
    return Dtype::Unknown;
}

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// ─── Minimal JSON parser ───────────────────────────────────────────────────
//
// Safetensors headers use a strict subset of JSON: top-level object whose
// values are either an object {dtype, shape, data_offsets} or the
// __metadata__ string->string object. We parse exactly that and reject
// anything that doesn't match. No nested escapes beyond \\, \", \n, \t are
// expected in tensor names; we handle them defensively anyway.

namespace {

struct Parser {
    const char* p;
    const char* end;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("safetensors: parse error: " + msg);
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    bool eat(char c) {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("expected '") + c + "'");
    }

    std::string parse_string() {
        skip_ws();
        if (p >= end || *p != '"') fail("expected string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                char e = p[1];
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    default:   fail("unsupported escape in string");
                }
                p += 2;
            } else {
                out += *p++;
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;
        return out;
    }

    int64_t parse_int() {
        skip_ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; ++p; }
        if (p >= end || *p < '0' || *p > '9') fail("expected integer");
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            ++p;
        }
        return neg ? -v : v;
    }

    // Skip an arbitrary JSON value (used for __metadata__ entries we don't
    // care about, and as a defensive fallback).
    void skip_value() {
        skip_ws();
        if (p >= end) fail("unexpected EOF");
        if (*p == '"') { (void)parse_string(); return; }
        if (*p == '{' || *p == '[') {
            char open = *p++, close = (open == '{') ? '}' : ']';
            int depth = 1;
            while (p < end && depth > 0) {
                if (*p == '"') { (void)parse_string(); continue; }
                if (*p == open) ++depth;
                else if (*p == close) --depth;
                ++p;
            }
            return;
        }
        // number / true / false / null
        while (p < end && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') ++p;
    }
};

struct Entry {
    Dtype dtype = Dtype::Unknown;
    std::vector<int64_t> shape;
    uint64_t off_start = 0;
    uint64_t off_end   = 0;
};

Entry parse_entry(Parser& ps) {
    Entry e;
    ps.expect('{');
    bool first = true;
    while (true) {
        ps.skip_ws();
        if (ps.eat('}')) break;
        if (!first) ps.expect(',');
        first = false;
        std::string key = ps.parse_string();
        ps.expect(':');
        if (key == "dtype") {
            std::string v = ps.parse_string();
            e.dtype = parse_dtype(v);
            if (e.dtype == Dtype::Unknown) {
                throw std::runtime_error("safetensors: unsupported dtype '" + v + "'");
            }
        } else if (key == "shape") {
            ps.expect('[');
            bool sfirst = true;
            while (true) {
                ps.skip_ws();
                if (ps.eat(']')) break;
                if (!sfirst) ps.expect(',');
                sfirst = false;
                e.shape.push_back(ps.parse_int());
            }
        } else if (key == "data_offsets") {
            ps.expect('[');
            e.off_start = static_cast<uint64_t>(ps.parse_int());
            ps.expect(',');
            e.off_end   = static_cast<uint64_t>(ps.parse_int());
            ps.expect(']');
        } else {
            // Unknown field — skip its value defensively.
            ps.skip_value();
        }
    }
    return e;
}

}  // namespace

// ─── mmap (platform) ───────────────────────────────────────────────────────

namespace {

struct Mapping {
    void* file = nullptr;
    void* mapping = nullptr;
    void* base = nullptr;
    std::size_t size = 0;
};

Mapping map_file(const std::string& path) {
    Mapping m;
#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("safetensors: cannot open '" + path + "'");
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) {
        CloseHandle(fh);
        throw std::runtime_error("safetensors: GetFileSizeEx failed for '" + path + "'");
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        throw std::runtime_error("safetensors: CreateFileMapping failed for '" + path + "'");
    }
    void* base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        CloseHandle(mh);
        CloseHandle(fh);
        throw std::runtime_error("safetensors: MapViewOfFile failed for '" + path + "'");
    }
    m.file    = fh;
    m.mapping = mh;
    m.base    = base;
    m.size    = static_cast<std::size_t>(sz.QuadPart);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("safetensors: cannot open '" + path + "'");
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("safetensors: fstat failed for '" + path + "'");
    }
    void* base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("safetensors: mmap failed for '" + path + "'");
    }
    m.file    = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
    m.mapping = nullptr;
    m.base    = base;
    m.size    = static_cast<std::size_t>(st.st_size);
#endif
    return m;
}

void unmap_file(Mapping& m) noexcept {
#ifdef _WIN32
    if (m.base)    UnmapViewOfFile(m.base);
    if (m.mapping) CloseHandle(m.mapping);
    if (m.file)    CloseHandle(m.file);
#else
    if (m.base && m.size) ::munmap(m.base, m.size);
    if (m.file) ::close(static_cast<int>(reinterpret_cast<intptr_t>(m.file)));
#endif
    m = Mapping{};
}

}  // namespace

// ─── File ──────────────────────────────────────────────────────────────────

File::~File() { release_(); }

File::File(File&& other) noexcept
    : file_(other.file_), mapping_(other.mapping_), base_(other.base_),
      file_size_(other.file_size_),
      tensors_(std::move(other.tensors_)), index_(std::move(other.index_)) {
    other.file_ = other.mapping_ = other.base_ = nullptr;
    other.file_size_ = 0;
}

File& File::operator=(File&& other) noexcept {
    if (this != &other) {
        release_();
        file_      = other.file_;
        mapping_   = other.mapping_;
        base_      = other.base_;
        file_size_ = other.file_size_;
        tensors_   = std::move(other.tensors_);
        index_     = std::move(other.index_);
        other.file_ = other.mapping_ = other.base_ = nullptr;
        other.file_size_ = 0;
    }
    return *this;
}

void File::release_() noexcept {
    Mapping m{file_, mapping_, base_, file_size_};
    unmap_file(m);
    file_ = mapping_ = base_ = nullptr;
    file_size_ = 0;
    tensors_.clear();
    index_.clear();
}

File File::open(const std::string& path) {
    Mapping m = map_file(path);
    if (m.size < 8) {
        unmap_file(m);
        throw std::runtime_error("safetensors: file too small: '" + path + "'");
    }
    const uint8_t* bytes = static_cast<const uint8_t*>(m.base);
    uint64_t header_size = 0;
    std::memcpy(&header_size, bytes, 8);  // little-endian; platforms we target are LE
    if (header_size > m.size - 8) {
        unmap_file(m);
        throw std::runtime_error("safetensors: header_size exceeds file");
    }
    const char* hdr_begin = reinterpret_cast<const char*>(bytes + 8);
    const char* hdr_end   = hdr_begin + header_size;
    const uint8_t* payload_base = bytes + 8 + header_size;
    std::size_t payload_size = m.size - 8 - header_size;

    File f;
    f.file_      = m.file;
    f.mapping_   = m.mapping;
    f.base_      = m.base;
    f.file_size_ = m.size;

    Parser ps{hdr_begin, hdr_end};
    try {
        ps.expect('{');
        bool first = true;
        while (true) {
            ps.skip_ws();
            if (ps.eat('}')) break;
            if (!first) ps.expect(',');
            first = false;
            std::string name = ps.parse_string();
            ps.expect(':');
            if (name == "__metadata__") {
                ps.skip_value();
                continue;
            }
            Entry e = parse_entry(ps);
            if (e.off_end < e.off_start || e.off_end > payload_size) {
                throw std::runtime_error(
                    "safetensors: tensor '" + name + "' has out-of-range data_offsets");
            }
            TensorView tv;
            tv.name   = name;
            tv.dtype  = e.dtype;
            tv.shape  = std::move(e.shape);
            tv.data   = payload_base + e.off_start;
            tv.nbytes = static_cast<std::size_t>(e.off_end - e.off_start);

            int64_t expected = tv.numel() * dtype_size_bytes(tv.dtype);
            if (expected != static_cast<int64_t>(tv.nbytes)) {
                throw std::runtime_error(
                    "safetensors: tensor '" + name +
                    "' nbytes/shape mismatch");
            }
            f.index_.emplace(tv.name, f.tensors_.size());
            f.tensors_.push_back(std::move(tv));
        }
    } catch (...) {
        // f's destructor will unmap.
        throw;
    }
    return f;
}

const TensorView* File::find(std::string_view name) const {
    auto it = index_.find(std::string(name));
    if (it == index_.end()) return nullptr;
    return &tensors_[it->second];
}

const TensorView& File::get(std::string_view name) const {
    const TensorView* v = find(name);
    if (!v) {
        throw std::runtime_error(
            "safetensors: no tensor named '" + std::string(name) + "'");
    }
    return *v;
}

// ─── Upload ────────────────────────────────────────────────────────────────

void upload(const TensorView& view, int rows, int cols, brotensor::Tensor& dst) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("safetensors::upload: rows/cols must be positive");
    }
    const std::size_t expected =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
        static_cast<std::size_t>(dtype_size_bytes(view.dtype));
    if (expected != view.nbytes) {
        throw std::runtime_error(
            "safetensors::upload: byte count mismatch for tensor '" + view.name + "'");
    }
    if (view.dtype == Dtype::F32) {
        dst = brotensor::Tensor::from_host(reinterpret_cast<const float*>(view.data), rows, cols);
    } else if (view.dtype == Dtype::F16) {
        dst = brotensor::Tensor::from_host_fp16(
            reinterpret_cast<const uint16_t*>(view.data), rows, cols);
    } else {
        throw std::runtime_error(
            std::string("safetensors::upload: unsupported dtype ") +
            dtype_name(view.dtype) + " for tensor '" + view.name + "'");
    }
}

void upload_fp16(const TensorView& view, int rows, int cols, brotensor::Tensor& dst) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("safetensors::upload_fp16: rows/cols must be positive");
    }
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t expected = n * static_cast<std::size_t>(dtype_size_bytes(view.dtype));
    if (expected != view.nbytes) {
        throw std::runtime_error(
            "safetensors::upload_fp16: byte count mismatch for tensor '" + view.name + "'");
    }
    if (view.dtype == Dtype::F16) {
        dst = brotensor::Tensor::from_host_fp16(
            reinterpret_cast<const uint16_t*>(view.data), rows, cols);
    } else if (view.dtype == Dtype::F32) {
        const float* src = reinterpret_cast<const float*>(view.data);
        std::vector<uint16_t> tmp(n);
        for (std::size_t i = 0; i < n; ++i) {
            tmp[i] = brotensor::fp32_to_fp16_bits(src[i]);
        }
        dst = brotensor::Tensor::from_host_fp16(tmp.data(), rows, cols);
    } else {
        throw std::runtime_error(
            std::string("safetensors::upload_fp16: unsupported dtype ") +
            dtype_name(view.dtype) + " for tensor '" + view.name + "'");
    }
}

// ─── Writer ────────────────────────────────────────────────────────────────

namespace {

const char* dtype_safetensors_name(Dtype d) {
    switch (d) {
        case Dtype::F32:  return "F32";
        case Dtype::F16:  return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::I32:  return "I32";
        case Dtype::I64:  return "I64";
        case Dtype::U8:   return "U8";
        case Dtype::BOOL: return "BOOL";
        default: throw std::runtime_error("safetensors::write: unsupported dtype");
    }
}

void json_escape_append(std::string& out, const std::string& s) {
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

}  // namespace

void write_file(const std::string& path, const std::vector<WriteEntry>& entries) {
    // Validate + compute offsets.
    std::vector<uint64_t> off_start(entries.size()), off_end(entries.size());
    uint64_t cursor = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const WriteEntry& e = entries[i];
        if (e.name.empty()) {
            throw std::runtime_error("safetensors::write: empty tensor name");
        }
        if (!e.host_data && e.bytes > 0) {
            throw std::runtime_error("safetensors::write: '" + e.name + "' null data");
        }
        int dsz = dtype_size_bytes(e.dtype);
        if (dsz <= 0) {
            throw std::runtime_error("safetensors::write: '" + e.name + "' bad dtype");
        }
        int64_t n = 1;
        for (int64_t d : e.shape) {
            if (d < 0) throw std::runtime_error("safetensors::write: '" + e.name + "' negative shape");
            n *= d;
        }
        std::size_t expected = static_cast<std::size_t>(n) * static_cast<std::size_t>(dsz);
        if (expected != e.bytes) {
            throw std::runtime_error("safetensors::write: '" + e.name +
                "' byte count mismatch (shape=" + std::to_string(expected) +
                " bytes=" + std::to_string(e.bytes) + ")");
        }
        off_start[i] = cursor;
        cursor += e.bytes;
        off_end[i]   = cursor;
    }

    // Build JSON header.
    std::string hdr;
    hdr.reserve(256 * entries.size() + 16);
    hdr += '{';
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i) hdr += ',';
        hdr += '"';
        json_escape_append(hdr, entries[i].name);
        hdr += "\":{\"dtype\":\"";
        hdr += dtype_safetensors_name(entries[i].dtype);
        hdr += "\",\"shape\":[";
        for (std::size_t j = 0; j < entries[i].shape.size(); ++j) {
            if (j) hdr += ',';
            hdr += std::to_string(entries[i].shape[j]);
        }
        hdr += "],\"data_offsets\":[";
        hdr += std::to_string(off_start[i]);
        hdr += ',';
        hdr += std::to_string(off_end[i]);
        hdr += "]}";
    }
    hdr += '}';
    // Pad header to 8-byte alignment so payload tensor starts are aligned.
    while ((hdr.size() % 8) != 0) hdr += ' ';

    uint64_t header_size = static_cast<uint64_t>(hdr.size());

    // Write file.
#ifdef _WIN32
    FILE* fp = nullptr;
    fopen_s(&fp, path.c_str(), "wb");
#else
    FILE* fp = std::fopen(path.c_str(), "wb");
#endif
    if (!fp) throw std::runtime_error("safetensors::write: cannot open '" + path + "'");

    auto wfail = [&](const std::string& what) {
        std::fclose(fp);
        throw std::runtime_error("safetensors::write: " + what + " '" + path + "'");
    };

    if (std::fwrite(&header_size, 1, 8, fp) != 8) wfail("header_size write failed");
    if (std::fwrite(hdr.data(), 1, hdr.size(), fp) != hdr.size()) wfail("header write failed");
    for (const WriteEntry& e : entries) {
        if (e.bytes == 0) continue;
        if (std::fwrite(e.host_data, 1, e.bytes, fp) != e.bytes) wfail("payload write failed");
    }
    std::fclose(fp);
}

}  // namespace brodiffusion::safetensors
