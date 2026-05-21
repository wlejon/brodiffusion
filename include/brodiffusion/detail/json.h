#pragma once

// Header-only general-purpose JSON parser.
//
// Handles nested objects, arrays, strings (with the standard
// \" \\ \/ \n \t \r \b \f \uXXXX escapes, including UTF-16 surrogate pairs),
// integers, floating-point with exponents, true, false and null. Numbers are
// stored internally as double (config files freely mix ints and floats).
//
// Used by brodiffusion::model_config (diffusers model_index.json / config.json
// files) and, in a later phase, by the HuggingFace tokenizer.json loader — so
// it aims to be correct and general, not config-specific.
//
// Recursive-descent. On malformed input, parse() throws std::runtime_error
// with a "json: ..." message that includes a byte offset.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::detail::json {

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
public:
    using Array  = std::vector<Value>;
    using Member = std::pair<std::string, Value>;
    using Object = std::vector<Member>;

    Value() : type_(Type::Null) {}

    static Value make_null()                 { return Value(); }
    static Value make_bool(bool b)           { Value v; v.type_ = Type::Bool;   v.bool_ = b;   return v; }
    static Value make_number(double d)       { Value v; v.type_ = Type::Number; v.num_  = d;   return v; }
    static Value make_string(std::string s)  { Value v; v.type_ = Type::String; v.str_  = std::move(s); return v; }
    static Value make_array(Array a)         { Value v; v.type_ = Type::Array;  v.arr_  = std::move(a);  return v; }
    static Value make_object(Object o)       { Value v; v.type_ = Type::Object; v.obj_  = std::move(o);  return v; }

    Type type() const { return type_; }
    bool is_null()   const { return type_ == Type::Null;   }
    bool is_bool()   const { return type_ == Type::Bool;   }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array()  const { return type_ == Type::Array;  }
    bool is_object() const { return type_ == Type::Object; }

    // Throwing typed accessors.
    double as_number() const {
        if (type_ != Type::Number) throw std::runtime_error("json: value is not a number");
        return num_;
    }
    bool as_bool() const {
        if (type_ != Type::Bool) throw std::runtime_error("json: value is not a bool");
        return bool_;
    }
    const std::string& as_string() const {
        if (type_ != Type::String) throw std::runtime_error("json: value is not a string");
        return str_;
    }
    const Array& as_array() const {
        if (type_ != Type::Array) throw std::runtime_error("json: value is not an array");
        return arr_;
    }
    const Object& as_object() const {
        if (type_ != Type::Object) throw std::runtime_error("json: value is not an object");
        return obj_;
    }

    // Object lookup.
    bool contains(const std::string& key) const { return find(key) != nullptr; }

    const Value* find(const std::string& key) const {
        if (type_ != Type::Object) return nullptr;
        for (const auto& m : obj_) {
            if (m.first == key) return &m.second;
        }
        return nullptr;
    }
    const Value& at(const std::string& key) const {
        const Value* v = find(key);
        if (!v) throw std::runtime_error("json: missing key '" + key + "'");
        return *v;
    }

    // Convenience getters with defaults. A key that is absent OR explicitly
    // JSON-null returns the default.
    int get_int(const std::string& key, int dflt) const {
        const Value* v = find(key);
        if (!v || v->is_null()) return dflt;
        return static_cast<int>(v->as_number());
    }
    float get_float(const std::string& key, float dflt) const {
        const Value* v = find(key);
        if (!v || v->is_null()) return dflt;
        return static_cast<float>(v->as_number());
    }
    bool get_bool(const std::string& key, bool dflt) const {
        const Value* v = find(key);
        if (!v || v->is_null()) return dflt;
        return v->as_bool();
    }
    std::string get_string(const std::string& key, const std::string& dflt) const {
        const Value* v = find(key);
        if (!v || v->is_null()) return dflt;
        return v->as_string();
    }
    std::vector<int> get_int_array(const std::string& key,
                                   const std::vector<int>& dflt) const {
        const Value* v = find(key);
        if (!v || v->is_null()) return dflt;
        const Array& a = v->as_array();
        std::vector<int> out;
        out.reserve(a.size());
        for (const auto& e : a) out.push_back(static_cast<int>(e.as_number()));
        return out;
    }

private:
    Type        type_;
    bool        bool_ = false;
    double      num_  = 0.0;
    std::string str_;
    Array       arr_;
    Object      obj_;
};

namespace detail {

// Recursive-descent parser over a contiguous text buffer.
class Parser {
public:
    explicit Parser(const std::string& text)
        : begin_(text.data()), p_(text.data()), end_(text.data() + text.size()) {}

    Value parse_document() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (p_ != end_) fail("trailing characters after document");
        return v;
    }

private:
    const char* begin_;
    const char* p_;
    const char* end_;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("json: " + msg + " at offset "
                                 + std::to_string(p_ - begin_));
    }

    void skip_ws() {
        while (p_ < end_ &&
               (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) {
            ++p_;
        }
    }

    char peek() const {
        if (p_ >= end_) return '\0';
        return *p_;
    }

    Value parse_value() {
        skip_ws();
        if (p_ >= end_) fail("unexpected end of input");
        switch (*p_) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return Value::make_string(parse_string());
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default:
                if (*p_ == '-' || (*p_ >= '0' && *p_ <= '9')) {
                    return parse_number();
                }
                fail("unexpected character");
        }
    }

    Value parse_object() {
        ++p_;  // consume '{'
        Value::Object members;
        skip_ws();
        if (peek() == '}') { ++p_; return Value::make_object(std::move(members)); }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expected object key string");
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expected ':' after object key");
            ++p_;
            Value val = parse_value();
            members.emplace_back(std::move(key), std::move(val));
            skip_ws();
            char c = peek();
            if (c == ',') { ++p_; continue; }
            if (c == '}') { ++p_; break; }
            fail("expected ',' or '}' in object");
        }
        return Value::make_object(std::move(members));
    }

    Value parse_array() {
        ++p_;  // consume '['
        Value::Array elems;
        skip_ws();
        if (peek() == ']') { ++p_; return Value::make_array(std::move(elems)); }
        while (true) {
            elems.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') { ++p_; continue; }
            if (c == ']') { ++p_; break; }
            fail("expected ',' or ']' in array");
        }
        return Value::make_array(std::move(elems));
    }

    // Append a Unicode codepoint to `out` as UTF-8.
    static void append_utf8(uint32_t cp, std::string& out) {
        if (cp <= 0x7F) {
            out += static_cast<char>(cp);
        } else if (cp <= 0x7FF) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t parse_hex4() {
        if (end_ - p_ < 4) fail("truncated \\u escape");
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            char c = p_[i];
            int v = (c >= '0' && c <= '9') ? c - '0'
                  : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                  : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                  : -1;
            if (v < 0) fail("bad hex digit in \\u escape");
            cp = (cp << 4) | static_cast<uint32_t>(v);
        }
        p_ += 4;
        return cp;
    }

    std::string parse_string() {
        if (peek() != '"') fail("expected string");
        ++p_;  // consume opening quote
        std::string out;
        while (p_ < end_) {
            char c = *p_;
            if (c == '"') { ++p_; return out; }
            if (c == '\\') {
                ++p_;
                if (p_ >= end_) fail("unterminated escape in string");
                char e = *p_++;
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        // UTF-16 surrogate pair handling.
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (end_ - p_ >= 2 && p_[0] == '\\' && p_[1] == 'u') {
                                p_ += 2;
                                uint32_t lo = parse_hex4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10)
                                                 + (lo - 0xDC00);
                                } else {
                                    // Unpaired high surrogate; emit both raw.
                                    append_utf8(cp, out);
                                    cp = lo;
                                }
                            }
                        }
                        append_utf8(cp, out);
                        break;
                    }
                    default: fail("unsupported escape sequence");
                }
            } else {
                out += c;
                ++p_;
            }
        }
        fail("unterminated string");
    }

    Value parse_number() {
        const char* start = p_;
        if (peek() == '-') ++p_;
        if (peek() == '0') {
            ++p_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++p_;
        } else {
            fail("invalid number");
        }
        if (peek() == '.') {
            ++p_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number: digits expected after '.'");
            while (peek() >= '0' && peek() <= '9') ++p_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++p_;
            if (peek() == '+' || peek() == '-') ++p_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number: digits expected in exponent");
            while (peek() >= '0' && peek() <= '9') ++p_;
        }
        std::string num(start, p_);
        try {
            return Value::make_number(std::stod(num));
        } catch (const std::exception&) {
            fail("number out of range");
        }
    }

    Value parse_bool() {
        if (end_ - p_ >= 4 && std::string(p_, p_ + 4) == "true") {
            p_ += 4;
            return Value::make_bool(true);
        }
        if (end_ - p_ >= 5 && std::string(p_, p_ + 5) == "false") {
            p_ += 5;
            return Value::make_bool(false);
        }
        fail("invalid literal");
    }

    Value parse_null() {
        if (end_ - p_ >= 4 && std::string(p_, p_ + 4) == "null") {
            p_ += 4;
            return Value::make_null();
        }
        fail("invalid literal");
    }
};

}  // namespace detail

// Parse a whole JSON document. Throws std::runtime_error on malformed input.
inline Value parse(const std::string& text) {
    detail::Parser parser(text);
    return parser.parse_document();
}

}  // namespace brodiffusion::detail::json
