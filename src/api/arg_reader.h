#pragma once

#include "embed/embed.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace brodiffusion::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

inline double numAt(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return 0.0;
    Value v = args[i];
    if (ev::isObject(v)) return 0.0;
    double d = ev::toDouble(v);
    return std::isnan(d) ? 0.0 : d;
}

inline int32_t i32At(std::span<const Value> args, size_t i) {
    return static_cast<int32_t>(static_cast<int64_t>(numAt(args, i)));
}

inline uint32_t u32At(std::span<const Value> args, size_t i) {
    return static_cast<uint32_t>(static_cast<int64_t>(numAt(args, i)));
}

inline int64_t i64At(std::span<const Value> args, size_t i) {
    return static_cast<int64_t>(numAt(args, i));
}

inline uint64_t u64At(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return 0;
    Value v = args[i];
    if (ev::isObject(v)) return 0;
    return ev::toUint64(v);
}

inline bool boolAt(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return false;
    return ev::toBool(args[i]);
}

inline std::string strAt(std::span<const Value> args, size_t i) {
    if (i >= args.size() || ev::isUndefined(args[i]) || ev::isNull(args[i]) || ev::isSymbol(args[i])) return "";
    return ev::toUtf8(args[i]);
}

inline Value argAt(std::span<const Value> args, size_t i) {
    return i < args.size() ? args[i] : ev::undefined();
}

inline bool hasArg(std::span<const Value> args, size_t i) {
    return i < args.size() && !ev::isUndefined(args[i]);
}

class ArgReader {
public:
    explicit ArgReader(std::span<const Value> args) : args_(args) {}

    double getDouble(size_t i, double def = 0.0) const {
        return hasArg(args_, i) ? numAt(args_, i) : def;
    }
    int getInt(size_t i, int def = 0) const {
        return hasArg(args_, i) ? i32At(args_, i) : def;
    }
    uint32_t getUint(size_t i, uint32_t def = 0) const {
        return hasArg(args_, i) ? u32At(args_, i) : def;
    }
    bool getBool(size_t i, bool def = false) const {
        return hasArg(args_, i) ? boolAt(args_, i) : def;
    }
    std::string getString(size_t i, const std::string& def = "") const {
        return hasArg(args_, i) ? strAt(args_, i) : def;
    }
    Value get(size_t i) const {
        return argAt(args_, i);
    }
    bool has(size_t i) const {
        return hasArg(args_, i);
    }
    size_t count() const {
        return args_.size();
    }

private:
    std::span<const Value> args_;
};

// ── option-bag readers ─────────────────────────────────────────────────────
// The old QuickJS bindings read an options object key by key, leaving the
// caller's default in place when the key is absent. These mirror getInt /
// getNum / getStr / getBool from diffusion_bindings.cpp so a port reads the
// same way.

inline bool propStr(Value obj, const char* key, std::string& dst) {
    if (!ev::isObject(obj)) return false;
    Value v = ev::getProperty(obj, key);
    if (!ev::isString(v)) return false;
    dst = ev::toUtf8(v);
    return true;
}

inline void propInt(Value obj, const char* key, int& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, key);
    if (ev::isNumber(v)) dst = static_cast<int>(ev::toDouble(v));
}

inline void propNum(Value obj, const char* key, float& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, key);
    if (ev::isNumber(v)) dst = static_cast<float>(ev::toDouble(v));
}

inline bool propBool(Value obj, const char* key, bool def = false) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    return ev::toBool(v);
}

// A plain JS number (lossless to 2^53) or a BigInt (full range).
inline void propSeed(Value obj, const char* key, uint64_t& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, key);
    if (ev::isBigInt(v)) dst = ev::toUint64(v);
    else if (ev::isNumber(v)) dst = static_cast<uint64_t>(ev::toInt64(v));
}

// ── arrays ─────────────────────────────────────────────────────────────────
// bronze::embed has no createArray; an empty array literal is the portable
// way to mint one. The accumulator is rooted because setElement allocates.

inline Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make) {
    ev::Persistent arr(ev::parseJson("[]").value);
    for (size_t i = 0; i < count; ++i) {
        ev::Persistent item(make(i));
        arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), item.get()));
    }
    return arr.get();
}

// Length of an array-like (JS `.length`), 0 for a non-object.
inline uint32_t arrayLength(Value v) {
    if (!ev::isObject(v)) return 0;
    Value len = ev::getProperty(v, "length");
    if (!ev::isNumber(len)) return 0;
    double d = ev::toDouble(len);
    return (d > 0.0 && d < 4294967295.0) ? static_cast<uint32_t>(d) : 0;
}

// Object.keys(obj): own enumerable string keys, for the option bags keyed by
// user-chosen names (the control-axis weight map).
inline std::vector<std::string> objectKeys(Value obj) {
    std::vector<std::string> out;
    if (!ev::isObject(obj)) return out;
    ev::Persistent root(obj);
    auto objectCtor = ev::globalValue("Object");
    if (!objectCtor.found || !ev::isObject(objectCtor.value)) return out;
    ev::Persistent keysFn(ev::getProperty(objectCtor.value, "keys"));
    if (!ev::isFunction(keysFn.get())) return out;
    const Value callArgs[1] = {root.get()};
    ev::CallResult r = ev::call(keysFn.get(), ev::undefined(), std::span<const Value>(callArgs, 1));
    if (r.thrown || !ev::isObject(r.value)) return out;
    ev::Persistent arr(r.value);
    const uint32_t n = arrayLength(arr.get());
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) out.push_back(ev::toUtf8(ev::getElement(arr.get(), i)));
    return out;
}

} // namespace brodiffusion::api
